# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""Bio-Logic MPT file reader with metadata extraction using traitlets.

Main classes:
- BiologicMPTReader: Modern reader with traitlets support for MPT files

Based on: https://github.com/echemdata/galvani/blob/master/galvani/BioLogic.py
"""

import contextlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import xarray as xr
from traitlets import HasTraits, Unicode
from traitlets import List as TList

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)

# --- Global Constants for Column Mapping ---

UNKNOWN_COLUMN_TYPE_HIERARCHY = ("<f8", "<f4", "<u4", "<u2", "<u1")

BOOL_COLUMNS = {
    "ox/red",
    "error",
    "control changes",
    "Ns changes",
    "counter inc.",
}

INT_COLUMNS = {"cycle number", "I Range", "Ns", "half cycle", "z cycle"}

FLOAT_COLUMNS = {
    "time/s",
    "P/W",
    "(Q-Qo)/mA.h",
    "x",
    "control/V",
    "control/mA",
    "control/V/mA",
    "(Q-Qo)/C",
    "dQ/C",
    "freq/Hz",
    "|Ewe|/V",
    "|I|/A",
    "Phase(Z)/deg",
    "|Z|/Ohm",
    "Re(Z)/Ohm",
    "-Im(Z)/Ohm",
    "Re(M)",
    "Im(M)",
    "|M|",
    "Re(Permittivity)",
    "Im(Permittivity)",
    "|Permittivity|",
    "Tan(Delta)",
    "Q charge/discharge/mA.h",
    "step time/s",
    "Q charge/mA.h",
    "Q discharge/mA.h",
    "Temperature/°C",
    "Efficiency/%",
    "Capacity/mA.h",
}

FLOAT_SUFFIXES = (
    "/s",
    "/Hz",
    "/deg",
    "/W",
    "/mW",
    "/W.h",
    "/mW.h",
    "/A",
    "/mA",
    "/A.h",
    "/mA.h",
    "/V",
    "/mV",
    "/F",
    "/mF",
    "/uF",
    "/µF",
    "/nF",
    "/C",
    "/Ohm",
    "/Ohm-1",
    "/Ohm.cm",
    "/mS/cm",
    "/%",
)

SPECIAL_MAPPINGS = {
    "dq/mA.h": ("dQ/mA.h", np.float64),
    "dQ/mA.h": ("dQ/mA.h", np.float64),
    "I/mA": ("I/mA", np.float64),
    "<I>/mA": ("I/mA", np.float64),
    "Ewe/V": ("Ewe/V", np.float64),
    "<Ewe>/V": ("Ewe/V", np.float64),
    "Ecell/V": ("Ewe/V", np.float64),
    "<Ewe/V>": ("Ewe/V", np.float64),
}


def _get_dtype_from_column_type(fieldname: str) -> Any:
    """Helper to get dtype based on column classification."""
    if fieldname in BOOL_COLUMNS:
        return np.bool_
    if fieldname in INT_COLUMNS:
        return np.int_
    if fieldname in FLOAT_COLUMNS:
        return np.float64
    if fieldname.endswith(FLOAT_SUFFIXES) or fieldname.startswith("empty_column_"):
        return np.float64
    return None


def fieldname_to_dtype(fieldname: str) -> Tuple[str, Any]:
    """Convert column header from MPT file to (name, dtype) tuple."""
    if fieldname == "mode":
        return ("mode", np.uint8)

    if fieldname in SPECIAL_MAPPINGS:
        return SPECIAL_MAPPINGS[fieldname]

    dtype = _get_dtype_from_column_type(fieldname)
    if dtype is not None:
        return (fieldname, dtype)

    raise ValueError(f"Invalid column header: {fieldname}")


def _calculate_systime(acq_start: str, relative_times: np.ndarray) -> pd.Series:
    """Calculate absolute system time from acquisition start and relative times."""
    try:
        # BioLogic format: MM/DD/YYYY HH:MM:SS.ffffff
        start_dt = datetime.strptime(acq_start, "%m/%d/%Y %H:%M:%S.%f")
        start_ts = start_dt.timestamp()
        return pd.Series(pd.to_datetime(start_ts + relative_times, unit="s"))
    except Exception as e:
        logger.debug("Failed to parse acquisition start time '%s': %s", acq_start, e)
        return pd.Series(relative_times)


def _read_mpt_content(mpt_file: Any, encoding: str = "latin1") -> Tuple[np.ndarray, List[bytes]]:
    """Internal helper to read MPT content from a file object."""
    magic = next(mpt_file).strip()
    if magic not in {b"EC-Lab ASCII FILE", b"BT-Lab ASCII FILE"}:
        raise ValueError(f"Bad first line: {magic!r}")

    nb_headers_match = re.match(rb"Nb header lines : (\d+)\s*$", next(mpt_file))
    if not nb_headers_match:
        raise ValueError("Invalid header line format")
    nb_headers = int(nb_headers_match.group(1))
    if nb_headers < 3:
        raise ValueError(f"Too few header lines: {nb_headers}")

    comments = [next(mpt_file) for _ in range(nb_headers - 3)]

    fieldnames_raw = next(mpt_file).decode(encoding).strip()
    fieldnames = fieldnames_raw.split("\t")

    current_pos = mpt_file.tell()
    first_data_line = next(mpt_file).decode(encoding).strip()
    mpt_file.seek(current_pos)
    data_column_count = len(first_data_line.split("\t"))

    if len(fieldnames) > data_column_count:
        fieldnames = fieldnames[:data_column_count]

    for i, fn in enumerate(fieldnames):
        if not fn or not fn.strip():
            fieldnames[i] = f"empty_column_{i}"

    record_type = np.dtype(list(map(fieldname_to_dtype, fieldnames)))

    def str_to_float(s: str) -> float:
        return float(s.replace(",", "."))

    converter_dict: Dict[int, Any] = dict.fromkeys(range(len(fieldnames)), str_to_float)
    mpt_array = np.loadtxt(mpt_file, dtype=record_type, converters=converter_dict)  # type: ignore[arg-type]

    return mpt_array, comments


# --- BiologicMPTReader Class ---


class BiologicMPTReader(HasTraits):
    """Reader for BioLogic MPT files with metadata extraction."""

    # --- Constants ---
    INSTRUMENT_NAME: ClassVar[str] = "BioLogic"
    DEFAULT_TECHNIQUE: ClassVar[List[str]] = ["echem"]
    MASS_REGEX: ClassVar[re.Pattern] = re.compile(r"(\d+\.?\d*)\s*(mg|g)", re.IGNORECASE)

    # --- Loader Metadata ---
    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "biologic"

    # --- Traitlets ---
    filepath = Unicode()
    active_material_mass = Unicode(allow_none=True)
    sample_name = Unicode(None, allow_none=True)
    start_time = Unicode(None, allow_none=True)
    operator = Unicode(None, allow_none=True)
    wave_number = Unicode(None, allow_none=True)
    technique = TList(Unicode(), default_value=DEFAULT_TECHNIQUE)
    # instrument traitlet removed to avoid conflict with ClassVar

    def __init__(self, filepath: Optional[Union[str, Path]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self, **_kwargs: Any) -> Tuple[RawData, RawDataInfo]:
        """Load BioLogic MPT file(s) and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath must be set before calling load()")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return self._load_single_file(path)
        if path.is_dir():
            return self._load_directory(path)

        raise ValueError(f"Path is neither a file nor a directory: {path}")

    def _load_single_file(self, path: Path) -> Tuple[RawData, RawDataInfo]:
        """Internal method to load a single BioLogic MPT file."""
        with open(path, "rb") as f:
            mpt_array, comments = _read_mpt_content(f)

        # Parse metadata
        file_info = self._parse_mpt_metadata(list(comments))
        metadata = {
            "file_info": file_info,
            "file_type": "MPT",
            "file_path": str(path),
        }
        cleaned_metadata = self._clean_metadata(metadata)

        # Determine mass
        mass = self._extract_mass(cleaned_metadata)

        # Detect techniques
        tech_list = self._detect_techniques(cleaned_metadata, mpt_array)

        # Create Dataset
        ds = self._create_dataset(mpt_array, cleaned_metadata, mass)

        raw_info = RawDataInfo(
            sample_name=self.sample_name or str(cleaned_metadata.get("sample_name", "Unknown")),
            start_time=self.start_time or cleaned_metadata.get("start_time"),
            operator=self.operator or cleaned_metadata.get("operator"),
            technique=self.technique if self.technique != self.DEFAULT_TECHNIQUE else tech_list,
            instrument=self.instrument,
            active_material_mass=self.active_material_mass or cleaned_metadata.get("active_material_mass"),
            wave_number=self.wave_number,
            others=cleaned_metadata,
        )

        return RawData(data=ds), raw_info

    def _extract_mass(self, metadata: Dict[str, Any]) -> Optional[float]:
        """Extract mass in grams from metadata or traitlet."""
        mass_str = self.active_material_mass or metadata.get("active_material_mass")
        if not mass_str:
            return None

        match = self.MASS_REGEX.search(str(mass_str))
        if match:
            val, unit = float(match.group(1)), match.group(2).lower()
            return val * 0.001 if unit == "mg" else val
        return None

    def _create_dataset(self, mpt_array: np.ndarray, metadata: Dict[str, Any], mass: Optional[float]) -> xr.Dataset:
        """Create a standardized xarray.Dataset from MPT array."""
        names = list(mpt_array.dtype.names or [])
        n_records = len(mpt_array)

        # Detect technique for column ordering
        tech_str = metadata.get("file_info", {}).get("technique", "")
        is_peis = "Electrochemical Impedance" in tech_str or "PEIS" in tech_str or "freq/Hz" in names
        is_gpcl = "Galvanostatic Cycling" in tech_str or "GPCL" in tech_str
        is_ocv = "Open Circuit Voltage" in tech_str or "OCV" in tech_str

        if is_peis:
            ordered_cols = ["cycle number", "freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm", "|Z|/Ohm", "Phase(Z)/deg"]
        elif is_gpcl:
            ordered_cols = ["time/s", "systime", "cycle number", "Ewe/V", "Ece/V", "voltage/V", "SpeCap_cal/mAh/g", "I/mA", "Capacity/mA.h"]
        elif is_ocv:
            ordered_cols = ["time/s", "systime", "cycle number", "Ewe/V", "Ece/V", "voltage/V"]
        else:
            ordered_cols = names

        data_vars = {col: (["record"], mpt_array[col]) for col in ordered_cols if col in names}
        coords = {"record": np.arange(1, n_records + 1)}

        # Add calculated columns
        extra_vars, extra_coords = self._compute_extra_columns(mpt_array, metadata, mass)
        data_vars.update(extra_vars)
        coords.update(extra_coords)

        # Final dataset with ordered columns
        ds = xr.Dataset({k: data_vars[k] for k in ordered_cols if k in data_vars}, coords=coords)
        self._apply_standard_attrs(ds)
        return ds

    @staticmethod
    def _compute_extra_columns(mpt_array: np.ndarray, metadata: Dict, mass: Optional[float]) -> Tuple[Dict, Dict]:
        """Compute additional columns like voltage, systime, and specific capacity."""
        extra_vars: Dict[str, Any] = {}
        extra_coords: Dict[str, Any] = {}
        names = mpt_array.dtype.names or []
        n_records = len(mpt_array)

        # Voltage
        # Ensure Ewe/V and Ece/V exist (set to 0 if missing as requested)
        if "Ewe/V" in names:
            ewe = mpt_array["Ewe/V"]
        else:
            ewe = np.zeros(n_records)
            extra_vars["Ewe/V"] = (["record"], ewe)

        if "Ece/V" in names:
            ece = mpt_array["Ece/V"]
        else:
            ece = np.zeros(n_records)
            extra_vars["Ece/V"] = (["record"], ece)

        extra_vars["voltage/V"] = (["record"], ewe - ece)

        # Time
        acq_start = metadata.get("file_info", {}).get("Acquisition started on", "")
        if acq_start and "time/s" in names:
            systimes = _calculate_systime(acq_start, mpt_array["time/s"])
            extra_coords["systime"] = (["record"], systimes)
            extra_coords["time_s"] = (["record"], (systimes - systimes[0]).dt.total_seconds())

        # Specific Capacity
        if mass and "Capacity/mA.h" in names:
            extra_vars["SpeCap_cal/mAh/g"] = (["record"], mpt_array["Capacity/mA.h"] / mass)

        return extra_vars, extra_coords

    @staticmethod
    def _apply_standard_attrs(ds: xr.Dataset) -> None:
        """Apply standard units and long names."""
        attr_map = {
            "time/s": {"units": "s", "long_name": "Time"},
            "Ewe/V": {"units": "V", "long_name": "Working Electrode Potential"},
            "Ece/V": {"units": "V", "long_name": "Counter Electrode Potential"},
            "I/mA": {"units": "mA", "long_name": "Current"},
            "voltage/V": {"units": "V", "long_name": "Cell Voltage"},
            "Capacity/mA.h": {"units": "mAh", "long_name": "Capacity"},
            "SpeCap_cal/mAh/g": {"units": "mAh/g", "long_name": "Specific Capacity"},
            "freq/Hz": {"units": "Hz", "long_name": "Frequency"},
            "Re(Z)/Ohm": {"units": "Ohm", "long_name": "Real Impedance"},
            "-Im(Z)/Ohm": {"units": "Ohm", "long_name": "Imaginary Impedance"},
        }
        for var, attrs in attr_map.items():
            if var in ds:
                ds[var].attrs.update(attrs)

    def _load_directory(self, path: Path) -> Tuple[RawData, RawDataInfo]:
        """Load all BioLogic MPT files in a directory into a DataTree."""
        mpt_files = sorted(path.rglob("*.mpt"))
        if not mpt_files:
            raise FileNotFoundError(f"No .mpt files found in {path}")

        tree_dict = {}
        infos = []

        for f in mpt_files:
            try:
                raw_data, raw_info = self._load_single_file(f)
                ds = cast(xr.Dataset, raw_data.data)

                # Sanitize for DataTree (no '/' allowed in variable names)
                # Build rename dict and drop conflicting variables
                rename_dict = {}
                vars_to_drop = []
                for v in ds.data_vars:
                    v_str = str(v)
                    if "/" in v_str:
                        new_name = v_str.replace("/", "_")
                        # Check if the new name already exists
                        if new_name not in ds:
                            rename_dict[v_str] = new_name
                        else:
                            # Drop the original variable with "/" since standardized version exists
                            vars_to_drop.append(v_str)

                if vars_to_drop:
                    ds = ds.drop_vars(vars_to_drop)
                if rename_dict:
                    ds = ds.rename(rename_dict)

                rel_path = f.relative_to(path).with_suffix("")
                node_path = "/" + "/".join(rel_path.parts)
                tree_dict[node_path] = ds
                infos.append(raw_info)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                continue

        if not tree_dict:
            raise RuntimeError(f"Failed to load any .mpt files from {path}")

        tree = xr.DataTree.from_dict(tree_dict, name=path.name)
        merged_info = self._merge_infos(infos, path)
        return RawData(data=tree), merged_info

    def _merge_infos(self, infos: List[RawDataInfo], root_path: Path) -> RawDataInfo:
        """Merge multiple RawDataInfo objects.

        Combines metadata from multiple files, storing unique values in lists.
        Removes duplicates from each field.
        """
        if not infos:
            return RawDataInfo()

        # Collect techniques from all files
        # 'echem' is kept unique, other techniques can repeat (one per file)
        all_techs = []
        seen_echem = False
        for info in infos:
            for tech in info.technique:
                if tech.lower() == "echem":
                    if not seen_echem:
                        all_techs.append(tech)
                        seen_echem = True
                else:
                    all_techs.append(tech)

        # Collect unique values for each field
        sample_names = sorted({info.sample_name for info in infos if info.sample_name})
        operators = sorted({info.operator for info in infos if info.operator})
        start_times = sorted({info.start_time for info in infos if info.start_time})
        masses = sorted({info.active_material_mass for info in infos if info.active_material_mass})
        wave_numbers = sorted({info.wave_number for info in infos if info.wave_number})

        # Build combined others dict with all unique metadata
        combined_others = {
            "n_files": len(infos),
            # Always store all sample names as a list for folder loading
            "sample_names": sample_names,
        }

        # Add list versions of other metadata if multiple unique values exist
        if len(operators) > 1:
            combined_others["all_operators"] = operators
        if len(masses) > 1:
            combined_others["all_active_material_masses"] = masses

        return RawDataInfo(
            sample_name=self.sample_name or root_path.name,  # Use folder name as primary sample_name
            technique=all_techs,
            instrument=self.instrument,
            operator=self.operator or (operators[0] if len(operators) == 1 else None),
            start_time=self.start_time or (start_times[0] if len(start_times) == 1 else None),
            active_material_mass=self.active_material_mass or (masses[0] if len(masses) == 1 else None),
            wave_number=self.wave_number or (wave_numbers[0] if len(wave_numbers) == 1 else None),
            others=combined_others,
        )

    def _detect_techniques(self, cleaned_metadata: Dict, mpt_array: np.ndarray) -> List[str]:
        """Detect specific electrochemical techniques."""
        tech_str = cleaned_metadata.get("file_info", {}).get("technique", "")
        names = mpt_array.dtype.names or []
        tech_list = list(self.technique)

        if "Electrochemical Impedance" in tech_str or "PEIS" in tech_str or "freq/Hz" in names:
            tech_list.append("peis")
        if "Galvanostatic Cycling" in tech_str or "GPCL" in tech_str:
            tech_list.append("gpcl")
        if "Open Circuit Voltage" in tech_str or "OCV" in tech_str:
            tech_list.append("ocv")

        return list(set(tech_list))

    @staticmethod
    def _parse_mpt_metadata(comments: List[Union[bytes, str]]) -> Dict[str, Any]:
        """Parse MPT file comments into structured metadata."""
        meta: Dict[str, Any] = {}
        state = {"current_section": None, "in_parameters": False, "work_mode_list": []}

        for line in comments:
            text = line.decode("latin1") if isinstance(line, bytes) else line
            text = text.rstrip("\r\n")
            if not text.strip():
                state["in_parameters"] = False
                continue

            BiologicMPTReader._handle_mpt_line(text, meta, state)

        if state["work_mode_list"]:
            meta["work_mode"] = state["work_mode_list"]
        return meta

    @staticmethod
    def _handle_mpt_line(text: str, meta: Dict, state: Dict) -> None:
        """Handle a single line of MPT metadata."""
        indent = len(text) - len(text.lstrip())
        content = text.strip()

        def split_kv(c: str) -> Optional[Tuple[str, str]]:
            for sep in (" : ", ":"):
                if sep in c:
                    k, v = c.split(sep, 1)
                    return k.strip(), v.strip()
            return None

        def add_val(d: Dict, key: str, val: Any) -> None:
            if key not in d:
                d[key] = val
            else:
                existing = d[key]
                if isinstance(existing, list):
                    existing.append(val)
                else:
                    d[key] = [existing, val]

        if indent > 0 and state["current_section"] is not None:
            kv = split_kv(content)
            if kv:
                add_val(state["current_section"], *kv)
        elif "technique" not in meta and any(kw in content.lower() for kw in ["electrochemical", "impedance", "spectroscopy", "potentio", "galvano", "open circuit", "ocv"]):
            meta["technique"] = content
        elif content.startswith("Cycle Definition"):
            state["in_parameters"] = True
            kv = split_kv(content)
            state["current_section"] = {"cycle_definition": kv[1]} if kv else {}
            state["work_mode_list"].append(state["current_section"])
        elif state["in_parameters"] and state["current_section"] is not None:
            match = re.match(r"(.+?)\s{2,}(.+)", text)
            if match:
                add_val(state["current_section"], match.group(1).strip(), match.group(2).strip())
            else:
                add_val(state["current_section"], content, "")
        else:
            kv = split_kv(content)
            if kv:
                k, v = kv
                if not v:
                    state["current_section"] = {}
                    meta[k] = state["current_section"]
                else:
                    add_val(meta, k, v)
                    state["current_section"] = None

    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to keep only essential fields."""
        cleaned: Dict[str, Any] = {}
        file_info = metadata.get("file_info", {})

        test_keys = ["technique", "Electrode material", "Electrolyte", "Mass of active material", "Reference electrode", "Acquisition started on", "Operator"]
        test_info = {k: file_info[k] for k in test_keys if k in file_info}

        if "Saved on" in file_info:
            saved = file_info["Saved on"]
            if isinstance(saved, dict):
                if "File" in saved:
                    test_info["name"] = saved["File"]
                if "Directory" in saved:
                    test_info["file_path"] = saved["Directory"]

        if test_info:
            cleaned.update({
                "sample_name": test_info.get("name"),
                "start_time": test_info.get("Acquisition started on"),
                "operator": test_info.get("Operator"),
                "active_material_mass": test_info.get("Mass of active material"),
            })

        proc_keys = ["Run on channel", "Ewe Ctrl range", "Electrode surface area", "Characteristic mass"]
        proc_info = {k: file_info[k] for k in proc_keys if k in file_info}
        if "Characteristic mass" in proc_info:
            cleaned.setdefault("active_material_mass", proc_info["Characteristic mass"])

        return {**cleaned, **metadata}
