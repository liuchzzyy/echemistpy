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
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

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
    header_lines: list[str] = field(default_factory=list)
    timestamp_string: str | None = None
    tstamp: float | None = None
    N_header_lines: int | None = None
    column_names: list[str] = field(default_factory=list)
    column_data: MutableMapping[str, list[float]] = field(default_factory=dict)
    ec_technique: str | None = None


class BiologicMPTReader:
    """Parse BioLogic ``.mpt`` and ``.mpr`` files into :class:`Measurement`.

    The implementation mirrors ixdat's reader but stores the parsed values in
    :mod:`xarray` containers so that the rest of echemistpy can work with the
    output directly. For binary ``.mpr`` files the class defers to the
    ``galvani`` project and includes a light-weight compatibility layer so that
    new column identifiers are automatically exposed instead of failing the
    parse.

    **Enhanced Information Extraction for .mpr files:**

    This reader now extracts comprehensive information from MPR files including:

    - All standard attributes (version, columns, dates, timestamps)
    - Module information (VMP settings, VMP data, VMP log)
    - VMP settings data with readable strings (battery capacity, technique names)
    - Flags dictionary for understanding data column meanings
    - Number of data points (npts)
    - Complete module metadata (names, versions, dates, sizes)

    The additional metadata is stored in the ``extras`` dictionary of the
    :class:`MeasurementMetadata` and can be accessed via:

    .. code-block:: python

        reader = BiologicMPTReader()
        measurement = reader.read("file.mpr")

        # Access enhanced metadata
        extras = measurement.metadata.extras
        print(extras["mpr_modules"])           # Module information
        print(extras["mpr_vmp_settings"])      # VMP settings
        print(extras["mpr_flags_dict"])        # Data flags meanings
        print(extras["mpr_npts"])              # Number of data points
    """

    def __init__(self):
        self.path_to_file: Path | None = None
        self.measurement_name: str | None = None
        self.state = _ReaderState()
        self.measurement: Measurement | None = None

    def read(
        self,
        path_to_file: str | Path,
        *,
        sample_name: str | None = None,
        instrument: str = "BioLogic EC-Lab",
        metadata_extras: Mapping[str, object] | None = None,
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

        suffix = self.path_to_file.suffix.lower()
        sample = sample_name or self.measurement_name

        if suffix == ".mpt":
            self._series_list_from_mpt()
            measurement = self._build_measurement(
                sample_name=sample,
                instrument=instrument,
                metadata_extras=metadata_extras,
            )
        elif suffix == ".mpr":
            measurement = self._build_measurement_from_mpr(
                sample_name=sample,
                instrument=instrument,
                metadata_extras=metadata_extras,
            )
        else:
            raise BiologicReadError(
                "Only BioLogic .mpt or .mpr exports are supported by this reader.",
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
        with open(self.path_to_file, encoding="ISO-8859-1") as handle:
            for line in handle:
                self._process_line(line)

        if t_str not in self.state.column_data:
            raise BiologicReadError(
                f"Missing mandatory time column '{t_str}' in {self.path_to_file}",
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
        # Pre-initialize all column lists to avoid repeated setdefault calls
        self.state.column_data = {name: [] for name in self.state.column_names}
        self.state.place_in_file = "data"

    def _process_data_line(self, line: str) -> None:
        data_strings_from_line = line.strip().split()
        for name, value_string in zip_longest(
            self.state.column_names,
            data_strings_from_line,
            fillvalue="0",
        ):
            parsed_value = self._parse_float(value_string, column=name)
            # Directly append without setdefault since dict is pre-initialized
            self.state.column_data[name].append(parsed_value)

    @staticmethod
    def _parse_float(value_string: str, *, column: str | None = None) -> float:
        try:
            return float(value_string)
        except ValueError:
            # Handle European decimal format (comma instead of period)
            if "," in value_string:
                try:
                    return float(value_string.replace(",", "."))
                except ValueError:
                    pass
            warnings.warn(
                f"Can't parse value string '{value_string}' in column '{column}'. Using 0",
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
        metadata_extras: Mapping[str, object] | None = None,
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
                    f"Skipping column '{column_name}' because it has {len(array)} samples while the time base has {n_rows}.",
                )
                continue
            data_vars[column_name] = xr.DataArray(
                array,
                dims=(dim_name,),
                attrs={"unit": get_column_unit_name(column_name)},
            )

        dataset = xr.Dataset(data_vars, coords=coords)
        dataset = dataset.assign_coords({t_str: (dim_name, time_values)})

        extras: dict[str, object] = {
            "ec_technique": self.state.ec_technique,
            "timestamp_string": self.state.timestamp_string,
            "tstamp": self.state.tstamp,
            "header": "".join(self.state.header_lines),
        }

        return self._finalize_measurement(
            dataset=dataset,
            time_values=time_values,
            sample_name=sample_name,
            instrument=instrument,
            metadata_extras=metadata_extras,
            extras=extras,
        )

    def _build_measurement_from_mpr(
        self,
        *,
        sample_name: str,
        instrument: str,
        metadata_extras: Mapping[str, object] | None,
    ) -> Measurement:
        dataset, extras, time_values = self._dataset_from_mpr()
        return self._finalize_measurement(
            dataset=dataset,
            time_values=time_values,
            sample_name=sample_name,
            instrument=instrument,
            metadata_extras=metadata_extras,
            extras=extras,
        )

    def _finalize_measurement(
        self,
        *,
        dataset: xr.Dataset,
        time_values: np.ndarray,
        sample_name: str,
        instrument: str,
        metadata_extras: Mapping[str, object] | None,
        extras: Mapping[str, object],
    ) -> Measurement:
        merged_extras = dict(extras)
        if metadata_extras:
            merged_extras.update(metadata_extras)

        metadata = MeasurementMetadata(
            technique="EC",
            sample_name=sample_name,
            instrument=instrument,
            extras=merged_extras,
            operator=None,
        )

        axis = Axis(name=t_str, unit="s", values=time_values)
        return Measurement(data=dataset, metadata=metadata, axes=[axis])

    def _dataset_from_mpr(self) -> tuple[xr.Dataset, dict[str, object], np.ndarray]:
        mpr_file = self._read_mpr_file()
        data = mpr_file.data
        column_names = data.dtype.names or ()
        if t_str not in column_names:
            raise BiologicReadError(
                f"Missing mandatory time column '{t_str}' in {self.path_to_file}",
            )

        dim_name = "time_index"
        coords = {dim_name: np.arange(data.shape[0], dtype=int)}
        data_vars = {}
        for column_name in column_names:
            array = np.asarray(data[column_name], dtype=float)
            data_vars[column_name] = xr.DataArray(
                array,
                dims=(dim_name,),
                attrs={"unit": get_column_unit_name(column_name)},
            )

        dataset = xr.Dataset(data_vars, coords=coords)
        time_values = np.asarray(data[t_str], dtype=float)
        dataset = dataset.assign_coords({t_str: (dim_name, time_values)})

        # Extract comprehensive metadata from mpr file
        extras: dict[str, object] = {
            "ec_technique": self.state.ec_technique,
            "timestamp_string": None,
            "tstamp": getattr(mpr_file, "timestamp", None),
            "header": None,
            "mpr_version": getattr(mpr_file, "version", None),
            "mpr_columns": tuple(int(value) for value in getattr(mpr_file, "cols", [])),
        }

        # Add basic MPR file attributes
        loop_index = getattr(mpr_file, "loop_index", None)
        if loop_index is not None:
            extras["mpr_loop_index"] = list(loop_index)

        start_date = getattr(mpr_file, "startdate", None)
        if start_date is not None:
            extras["mpr_start_date"] = str(start_date)

        end_date = getattr(mpr_file, "enddate", None)
        if end_date is not None:
            extras["mpr_end_date"] = str(end_date)

        # Add number of data points
        npts = getattr(mpr_file, "npts", None)
        if npts is not None:
            extras["mpr_npts"] = list(npts) if hasattr(npts, "__iter__") else [npts]

        # Extract flags dictionary for understanding data flags
        flags_dict = getattr(mpr_file, "flags_dict", None)
        if flags_dict is not None:
            extras["mpr_flags_dict"] = dict(flags_dict)

        # Extract information from modules
        modules_info = []
        vmp_settings = None
        vmp_log_data = None

        if hasattr(mpr_file, "modules") and mpr_file.modules:
            for module in mpr_file.modules:
                module_info = {
                    "shortname": module.get("shortname", b"").decode("utf-8", errors="ignore").strip(),
                    "longname": module.get("longname", b"").decode("utf-8", errors="ignore").strip(),
                    "version": module.get("version", None),
                    "date": module.get("date", b"").decode("utf-8", errors="ignore").strip(),
                    "length": module.get("length", None),
                }
                modules_info.append(module_info)

                # Extract VMP settings if available
                if module.get("shortname") == b"VMP Set   " and "data" in module:
                    vmp_settings = self._extract_vmp_settings(module["data"])

                # Extract VMP log data if available
                if module.get("shortname") == b"VMP LOG   " and "data" in module:
                    vmp_log_data = self._extract_vmp_log_data(module["data"])

        if modules_info:
            extras["mpr_modules"] = modules_info

        if vmp_settings:
            extras["mpr_vmp_settings"] = vmp_settings

        if vmp_log_data:
            extras["mpr_vmp_log"] = vmp_log_data

        timestamp = extras["tstamp"]
        if timestamp is not None:
            extras["tstamp"] = float(timestamp.timestamp())

        return dataset, extras, time_values

    def _read_mpr_file(self):
        try:
            from galvani import BioLogic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BiologicReadError(
                "Reading .mpr files requires the optional 'galvani' package. Install it to enable binary BioLogic parsing.",
            ) from exc

        if self.path_to_file is None:
            raise BiologicReadError("No file path provided for BioLogic reader.")

        patched_ids: set[int] = set()
        while True:
            try:
                with open(self.path_to_file, "rb") as handle:
                    return BioLogic.MPRfile(handle)
            except NotImplementedError as exc:
                column_id = self._extract_column_id_from_error(exc)
                if column_id is None:
                    raise BiologicReadError(
                        f"galvani could not parse {self.path_to_file}: {exc}",
                    ) from exc
                if column_id in patched_ids:
                    raise BiologicReadError(
                        f"Repeated failure while registering column ID {column_id} for {self.path_to_file}",
                    ) from exc
                self._register_unknown_column(column_id, BioLogic)
                patched_ids.add(column_id)
            except FileNotFoundError as exc:
                raise BiologicReadError(
                    f"Cannot open BioLogic file {self.path_to_file}: {exc}",
                ) from exc
            except Exception as exc:  # pragma: no cover - defensive guard
                raise BiologicReadError(
                    f"Unable to parse BioLogic file {self.path_to_file}: {exc}",
                ) from exc

    def _extract_vmp_settings(self, data: bytes) -> dict[str, object]:
        """Extract human-readable information from VMP settings data."""
        settings = {}

        # Try to extract readable strings from the binary data
        text_strings = []
        current_text = b""

        for byte in data:
            if 32 <= byte <= 126:  # Printable ASCII characters
                current_text += bytes([byte])
            else:
                if len(current_text) >= 3:  # Only keep strings of 3+ characters
                    text_strings.append(current_text.decode("ascii"))
                current_text = b""

        # Add final string if it exists
        if len(current_text) >= 3:
            text_strings.append(current_text.decode("ascii"))

        # Store all readable strings
        if text_strings:
            settings["readable_strings"] = text_strings

            # Try to identify common settings
            for text in text_strings:
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in ["a.h", "mah", "wh"]):
                    settings["battery_capacity"] = text.strip()
                elif any(keyword in text_lower for keyword in ["gitt", "gcpl", "gcps", "oca", "eis"]):
                    settings["technique_name"] = text.strip()
                elif "unspecified" in text_lower:
                    settings["reference_electrode"] = text.strip()

        return settings

    def _extract_vmp_log_data(self, data: bytes) -> dict[str, object]:
        """Extract information from VMP log data."""
        log_info = {}

        # Try to decode log data as text
        try:
            # Try different encodings
            for encoding in ["utf-8", "ascii", "latin-1"]:
                try:
                    text_data = data.decode(encoding, errors="ignore")
                    if len(text_data.strip()) > 0:
                        log_info["log_text"] = text_data.strip()
                        break
                except UnicodeDecodeError:
                    continue
        except Exception as exc:
            # Log decoding error but continue - we'll still extract size info
            warnings.warn(f"Could not decode VMP log data: {exc}", stacklevel=2)

        # If we can't decode as text, at least store the size
        log_info["log_size_bytes"] = len(data)

        return log_info

    @staticmethod
    def _extract_column_id_from_error(error: Exception) -> int | None:
        match = re.search(r"Column ID (\d+)", str(error))
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _register_unknown_column(column_id: int, module) -> None:
        name = f"unknown_{column_id}"
        if column_id not in module.VMPdata_colID_dtype_map:
            warnings.warn(
                f"Encountered unknown BioLogic column ID {column_id}. Treating it as a floating-point trace.",
                stacklevel=2,
            )
            module.VMPdata_colID_dtype_map[column_id] = (name, "<f4")

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


def fix_we_potential(
    measurement: Measurement,
    *,
    ewe_key: str = "<Ewe>/V",
    ece_key: str = "<Ece>/V",
    cell_key: str = "Ewe-Ece/V",
) -> Measurement:
    """Fix columns of zeros in ``<Ewe>/V`` for chronopotentiometry exports.

    The function is a direct adaptation of :func:`ixdat.readers.biologic.fix_we_potential`
    but operates on echemistpy's :class:`Measurement` containers.
    """

    measurement.require_variables([ewe_key, ece_key, cell_key])
    data = measurement.data
    dims = data[ewe_key].dims
    measurement.data = data.assign({ewe_key: (dims, data[cell_key].values + data[ece_key].values)})
    return measurement


def get_column_unit_name(column_name: str) -> str | None:
    """Return the unit name of a column, i.e. the substring after ``/``."""

    if "/" in column_name:
        return column_name.rsplit("/", maxsplit=1)[-1]
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
    "fix_we_potential",
    "get_column_unit_name",
    "timestamp_string_to_tstamp",
]
