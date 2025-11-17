"""BioLogic EC-Lab reader helpers.

This module ports the open-source implementation from ixdat's
``ixdat.readers.biologic`` module
(https://github.com/liuchzzyy/ixdat/blob/main/src/ixdat/readers/biologic.py)
so that echemistpy users can build on the same parsing logic while keeping the
original project attribution in place.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import xarray as xr

from echemistpy.io.structures import Axis, Measurement, MeasurementMetadata


delim = "\t"
t_str = "time/s"


class BiologicReadError(RuntimeError):
    """Raised when a BioLogic file cannot be parsed."""


@dataclass
class _ReaderState:
    """Internal bookkeeping so the parser can be reused for multiple files."""

    n_line: int = 0
    place_in_file: str = "header"
    header_lines: List[str] = field(default_factory=list)
    timestamp_string: Optional[str] = None
    tstamp: Optional[float] = None
    N_header_lines: Optional[int] = None
    column_names: List[str] = field(default_factory=list)
    column_data: MutableMapping[str, List[float]] = field(default_factory=dict)
    ec_technique: Optional[str] = None


class BiologicMPTReader:
    """Parse BioLogic ``.mpt`` files into :class:`~echemistpy.io.structures.Measurement`.

    The implementation mirrors ixdat's reader but stores the parsed values in
    :mod:`xarray` containers so that the rest of echemistpy can work with the
    output directly.
    """

    def __init__(self):
        self.path_to_file: Optional[Path] = None
        self.measurement_name: Optional[str] = None
        self.state = _ReaderState()
        self.measurement: Optional[Measurement] = None

    def read(
        self,
        path_to_file: str | Path,
        *,
        sample_name: Optional[str] = None,
        instrument: str = "BioLogic EC-Lab",
        metadata_extras: Optional[Mapping[str, object]] = None,
    ) -> Measurement:
        """Read ``path_to_file`` and return a :class:`Measurement`.

        Args:
            path_to_file: Absolute or relative path to the ``.mpt`` file.
            sample_name: Override the default sample name (derived from the file
                name).
            instrument: Stored in :class:`MeasurementMetadata`.
            metadata_extras: Optional additional metadata to merge with the
                information parsed from the file header.
        """

        self._reset_state()
        self.path_to_file = Path(path_to_file)
        self.measurement_name = self.path_to_file.name

        if self.path_to_file.suffix.lower() != ".mpt":
            raise BiologicReadError(
                "Only text-based .mpt exports are supported by this lightweight "
                "reader."
            )

        self._series_list_from_mpt()

        measurement = self._build_measurement(
            sample_name=sample_name or self.measurement_name,
            instrument=instrument,
            metadata_extras=metadata_extras,
        )
        self.measurement = measurement
        return measurement

    def _reset_state(self) -> None:
        self.state = _ReaderState()

    # ------------------------------------------------------------------
    # Parsing helpers (ported from ixdat)
    # ------------------------------------------------------------------
    def _series_list_from_mpt(self) -> None:
        assert self.path_to_file is not None
        with open(self.path_to_file, "r", encoding="ISO-8859-1") as handle:
            for line in handle:
                self._process_line(line)

        if t_str not in self.state.column_data:
            raise BiologicReadError(
                f"Missing mandatory time column '{t_str}' in {self.path_to_file}"
            )

    def _process_line(self, line: str) -> None:
        if self.state.place_in_file == "header":
            self._process_header_line(line)
        elif self.state.place_in_file == "column names":
            self._process_column_line(line)
        elif self.state.place_in_file == "data":
            self._process_data_line(line)
        else:
            raise BiologicReadError(f"Unexpected parser state: {self.state.place_in_file}")
        self.state.n_line += 1

    def _process_header_line(self, line: str) -> None:
        self.state.header_lines.append(line)
        if not self.state.N_header_lines:
            N_head_match = re.search(regular_expressions["N_header_lines"], line)
            if N_head_match:
                self.state.N_header_lines = int(N_head_match.group(1))
                return
        if self.state.n_line == 3:
            self.state.ec_technique = line.strip()
            return
        if not self.state.timestamp_string:
            timestamp_match = re.search(regular_expressions["timestamp_string"], line)
            if timestamp_match:
                self.state.timestamp_string = timestamp_match.group(1)
                self.state.tstamp = timestamp_string_to_tstamp(
                    self.state.timestamp_string,
                    forms=BIOLOGIC_TIMESTAMP_FORMS,
                )
                return
        loop_match = re.search(regular_expressions["loop"], line)
        if loop_match:
            n = int(loop_match.group(1))
            start = int(loop_match.group(2))
            finish = int(loop_match.group(3))
            self.state.column_data.setdefault("loop_number", [])
            self.state.column_data["loop_number"].extend([n] * (finish - start + 1))
            return

        if self.state.N_header_lines and self.state.n_line >= self.state.N_header_lines - 2:
            self.state.place_in_file = "column names"

    def _process_column_line(self, line: str) -> None:
        self.state.header_lines.append(line)
        self.state.column_names = line.strip().split(delim)
        self.state.column_data.update({name: [] for name in self.state.column_names})
        self.state.place_in_file = "data"

    def _process_data_line(self, line: str) -> None:
        data_strings_from_line = line.strip().split()
        for name, value_string in zip_longest(
            self.state.column_names, data_strings_from_line, fillvalue="0"
        ):
            parsed_value = self._parse_float(value_string, column=name)
            self.state.column_data.setdefault(name, []).append(parsed_value)

    @staticmethod
    def _parse_float(value_string: str, *, column: Optional[str] = None) -> float:
        try:
            return float(value_string)
        except ValueError:
            if "," in value_string:
                return BiologicMPTReader._parse_float(
                    value_string.replace(",", "."), column=column
                )
            warnings.warn(
                f"Can't parse value string '{value_string}' in column '{column}'. Using 0"
            )
            return 0.0

    # ------------------------------------------------------------------
    # Measurement creation
    # ------------------------------------------------------------------
    def _build_measurement(
        self,
        *,
        sample_name: str,
        instrument: str,
        metadata_extras: Optional[Mapping[str, object]] = None,
    ) -> Measurement:
        time_values = np.asarray(self.state.column_data[t_str], dtype=float)
        n_rows = len(time_values)
        if n_rows == 0:
            raise BiologicReadError("The measurement does not contain any time samples.")

        dim_name = "time_index"
        coords = {dim_name: np.arange(n_rows, dtype=int)}
        data_vars = {}
        for column_name, values in self.state.column_data.items():
            array = np.asarray(values, dtype=float)
            if len(array) != n_rows:
                warnings.warn(
                    f"Skipping column '{column_name}' because it has {len(array)} samples "
                    f"while the time base has {n_rows}."
                )
                continue
            data_vars[column_name] = xr.DataArray(
                array,
                dims=(dim_name,),
                attrs={"unit": get_column_unit_name(column_name)},
            )

        dataset = xr.Dataset(data_vars, coords=coords)
        dataset = dataset.assign_coords({t_str: (dim_name, time_values)})

        extras: Dict[str, object] = {
            "ec_technique": self.state.ec_technique,
            "timestamp_string": self.state.timestamp_string,
            "tstamp": self.state.tstamp,
            "header": "".join(self.state.header_lines),
        }
        if metadata_extras:
            extras.update(metadata_extras)

        metadata = MeasurementMetadata(
            technique="EC",
            sample_name=sample_name,
            instrument=instrument,
            extras=extras,
            operator=None,
        )

        axis = Axis(name=t_str, unit="s", values=time_values)
        return Measurement(data=dataset, metadata=metadata, axes=[axis])

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def print_header(self) -> None:
        """Print the raw header captured during parsing."""

        if not self.state.header_lines:
            raise BiologicReadError("No file has been parsed yet.")
        print("".join(self.state.header_lines))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"BiologicMPTReader({self.path_to_file})"


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------

def fix_WE_potential(
    measurement: Measurement,
    *,
    ewe_key: str = "<Ewe>/V",
    ece_key: str = "<Ece>/V",
    cell_key: str = "Ewe-Ece/V",
) -> Measurement:
    """Fix columns of zeros in ``<Ewe>/V`` for chronopotentiometry exports.

    The function is a direct adaptation of :func:`ixdat.readers.biologic.fix_WE_potential`
    but operates on echemistpy's :class:`Measurement` containers.
    """

    measurement.require_variables([ewe_key, ece_key, cell_key])
    data = measurement.data
    dims = data[ewe_key].dims
    measurement.data = data.assign({ewe_key: (dims, data[cell_key].values + data[ece_key].values)})
    return measurement


def get_column_unit_name(column_name: str) -> Optional[str]:
    """Return the unit name of a column, i.e. the substring after ``/``."""

    if "/" in column_name:
        return column_name.split("/")[-1]
    return None


def timestamp_string_to_tstamp(timestamp: str, *, forms: Sequence[str]) -> float:
    """Convert an EC-Lab timestamp string to a POSIX timestamp."""

    for form in forms:
        try:
            return datetime.strptime(timestamp, form).timestamp()
        except ValueError:
            continue
    raise BiologicReadError(f"Timestamp '{timestamp}' does not match known BioLogic formats.")


regular_expressions = {
    "N_header_lines": "Nb header lines : (.+)\n",
    "timestamp_string": "Acquisition started on : (.+)\n",
    "loop": "Loop ([0-9]+) from point number ([0-9]+) to ([0-9]+)",
}

BIOLOGIC_TIMESTAMP_FORMS = (
    "%m-%d-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%m-%d-%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H.%M.%S",
    "%m/%d/%Y %H.%M.%S.%f",
)

BIOLOGIC_COLUMN_NAMES = (
    "mode",
    "ox/red",
    "error",
    "control changes",
    "time/s",
    "control/V",
    "Ewe/V",
    "<I>/mA",
    "(Q-Qo)/C",
    "P/W",
    "loop number",
    "I/mA",
    "control/mA",
    "Ns changes",
    "counter inc.",
    "cycle number",
    "Ns",
    "(Q-Qo)/mA.h",
    "dQ/C",
    "Q charge/discharge/mA.h",
    "half cycle",
    "Capacitance charge/µF",
    "Capacitance discharge/µF",
    "dq/mA.h",
    "Q discharge/mA.h",
    "Q charge/mA.h",
    "Capacity/mA.h",
    "file number",
    "file_number",
    "Ece/V",
    "Ewe-Ece/V",
    "<Ece>/V",
    "<Ewe>/V",
    "Energy charge/W.h",
    "Energy discharge/W.h",
    "Efficiency/%",
    "Rcmp/Ohm",
)


__all__ = [
    "BIOLOGIC_COLUMN_NAMES",
    "BIOLOGIC_TIMESTAMP_FORMS",
    "BiologicMPTReader",
    "BiologicReadError",
    "fix_WE_potential",
    "get_column_unit_name",
    "timestamp_string_to_tstamp",
]
