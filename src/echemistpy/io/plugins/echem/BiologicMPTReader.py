# -*- coding: utf-8 -*-
"""Bio-Logic MPT file reader with metadata extraction using traitlets.

Main classes:
- BiologicDataReader: Modern reader with traitlets support for MPT files
- BiologicMPTReader: Legacy reader interface

Based on: https://github.com/echemdata/galvani/blob/master/galvani/BioLogic.py
"""
# ruff: noqa: N999

import logging
import re
import time
from collections import OrderedDict, defaultdict
from datetime import date, datetime
from os import SEEK_SET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from traitlets import HasTraits, List, Unicode

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)

UNKNOWN_COLUMN_TYPE_HIERARCHY = ("<f8", "<f4", "<u4", "<u2", "<u1")


def _get_dtype_from_column_type(fieldname: str) -> Any:
    """Helper to get dtype based on column classification."""
    # Boolean columns
    bool_columns = {
        "ox/red",
        "error",
        "control changes",
        "Ns changes",
        "counter inc.",
    }

    # Integer columns
    int_columns = {"cycle number", "I Range", "Ns", "half cycle", "z cycle"}

    # Float64 columns
    float_columns = {
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

    # Suffix-based float columns
    float_suffixes = {
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
    }

    if fieldname in bool_columns:
        return np.bool_
    if fieldname in int_columns:
        return np.int_
    if fieldname in float_columns:
        return np.float64
    if fieldname.endswith(tuple(float_suffixes)) or fieldname.startswith("empty_column_"):
        return np.float64
    return None


def fieldname_to_dtype(fieldname: str) -> tuple[str, Any]:
    """Convert column header from MPT file to (name, dtype) tuple."""
    # Special case: mode is uint8
    if fieldname == "mode":
        return ("mode", np.uint8)

    # Special mappings (field renames)
    special_mappings = {
        "dq/mA.h": ("dQ/mA.h", np.float64),
        "dQ/mA.h": ("dQ/mA.h", np.float64),
        "I/mA": ("I/mA", np.float64),
        "<I>/mA": ("I/mA", np.float64),
        "Ewe/V": ("Ewe/V", np.float64),
        "<Ewe>/V": ("Ewe/V", np.float64),
        "Ecell/V": ("Ewe/V", np.float64),
        "<Ewe/V>": ("Ewe/V", np.float64),
    }

    if fieldname in special_mappings:
        return special_mappings[fieldname]

    dtype = _get_dtype_from_column_type(fieldname)
    if dtype is not None:
        return (fieldname, dtype)

    raise ValueError("Invalid column header: %s" % fieldname)


def comma_converter(float_text: bytes) -> float:
    """Convert text to float, handling ',' as decimal point."""
    return float(float_text.translate(bytes.maketrans(b",", b".")))


def _calculate_systime(acq_start: str, relative_times: np.ndarray) -> pd.Series:
    """Calculate absolute system time from acquisition start and relative times.

    Args:
        acq_start: Acquisition start time string (e.g., "10/25/2022 13:57:09.123")
        relative_times: Array of relative times in seconds

    Returns:
        pandas.Series of datetime64[ns]
    """
    try:
        # BioLogic format: MM/DD/YYYY HH:MM:SS.ffffff
        start_dt = datetime.strptime(acq_start, "%m/%d/%Y %H:%M:%S.%f")
        start_ts = start_dt.timestamp()
        return pd.Series(pd.to_datetime(start_ts + relative_times, unit="s"))
    except Exception as e:
        logger.debug("Failed to parse acquisition start time '%s': %s", acq_start, e)
        # Fallback to just relative times if parsing fails
        return pd.Series(relative_times)


def mpt_file(file_or_path: str | Path, encoding: str = "latin1") -> tuple[np.ndarray, list[bytes]]:
    """Open .mpt files as numpy record arrays with comments."""
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as mpt_file:
            return _read_mpt_content(mpt_file, encoding)
    else:
        return _read_mpt_content(file_or_path, encoding)


def _read_mpt_content(mpt_file: Any, encoding: str = "latin1") -> tuple[np.ndarray, list[bytes]]:
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
        """Convert str to float, handling ',' as decimal point."""
        return float(s.replace(",", "."))

    converter_dict: dict[int, Any] = dict.fromkeys(range(len(fieldnames)), str_to_float)
    mpt_array = np.loadtxt(mpt_file, dtype=record_type, converters=converter_dict)  # type: ignore[arg-type]

    return mpt_array, comments


VMPmodule_hdr_v1 = np.dtype([
    ("shortname", "S10"),
    ("longname", "S25"),
    ("length", "<u4"),
    ("version", "<u4"),
    ("date", "S8"),
])

VMPmodule_hdr_v2 = np.dtype([
    ("shortname", "S10"),
    ("longname", "S25"),
    ("max length", "<u4"),
    ("length", "<u4"),
    ("version", "<u4"),
    ("unknown2", "<u4"),
    ("date", "S8"),
])


# Maps from colID to a tuple defining a numpy dtype
VMPdata_colID_dtype_map = {
    4: ("time/s", "<f8"),
    5: ("control/V/mA", "<f4"),
    6: ("Ewe/V", "<f4"),
    7: ("dq/mA.h", "<f8"),
    8: ("I/mA", "<f4"),  # 8 is either I or <I> ??
    9: ("Ece/V", "<f4"),
    11: ("<I>/mA", "<f8"),
    13: ("(Q-Qo)/mA.h", "<f8"),
    16: ("Analog IN 1/V", "<f4"),
    17: ("Analog IN 2/V", "<f4"),  # Probably column 18 is Analog IN 3/V, if anyone hits this error in the future
    19: ("control/V", "<f4"),
    20: ("control/mA", "<f4"),
    23: ("dQ/mA.h", "<f8"),  # Same as 7?
    24: ("cycle number", "<f8"),
    26: ("Rapp/Ohm", "<f4"),
    27: ("Ewe-Ece/V", "<f4"),
    32: ("freq/Hz", "<f4"),
    33: ("|Ewe|/V", "<f4"),
    34: ("|I|/A", "<f4"),
    35: ("Phase(Z)/deg", "<f4"),
    36: ("|Z|/Ohm", "<f4"),
    37: ("Re(Z)/Ohm", "<f4"),
    38: ("-Im(Z)/Ohm", "<f4"),
    39: ("I Range", "<u2"),
    69: ("R/Ohm", "<f4"),
    70: ("P/W", "<f4"),
    74: ("|Energy|/W.h", "<f8"),
    75: ("Analog OUT/V", "<f4"),
    76: ("<I>/mA", "<f4"),
    77: ("<Ewe>/V", "<f4"),
    78: ("Cs-2/µF-2", "<f4"),
    96: ("|Ece|/V", "<f4"),
    98: ("Phase(Zce)/deg", "<f4"),
    99: ("|Zce|/Ohm", "<f4"),
    100: ("Re(Zce)/Ohm", "<f4"),
    101: ("-Im(Zce)/Ohm", "<f4"),
    123: ("Energy charge/W.h", "<f8"),
    124: ("Energy discharge/W.h", "<f8"),
    125: ("Capacitance charge/µF", "<f8"),
    126: ("Capacitance discharge/µF", "<f8"),
    131: ("Ns", "<u2"),
    163: ("|Estack|/V", "<f4"),
    168: ("Rcmp/Ohm", "<f4"),
    169: ("Cs/µF", "<f4"),
    172: ("Cp/µF", "<f4"),
    173: ("Cp-2/µF-2", "<f4"),
    174: ("<Ewe>/V", "<f4"),
    # TODO: Unknown column IDs found in Biologic_EIS.mpr sample file
    # These appear to be EIS-related measurements but exact meaning is unknown
    # Source: 官方源信息和数据/Biologic_EIS.mpr
    175: ("unknown_175", "<f4"),  # Unknown EIS-related column
    176: ("unknown_176", "<f4"),  # Unknown EIS-related column
    177: ("unknown_177", "<f4"),  # Unknown EIS-related column
    178: ("(Q-Qo)/C", "<f4"),
    179: ("dQ/C", "<f4"),
    182: ("step time/s", "<f8"),
    211: ("Q charge/discharge/mA.h", "<f8"),
    212: ("half cycle", "<u4"),
    213: ("z cycle", "<u4"),
    # TODO: Unknown column ID found in Biologic_EIS.mpr sample file
    # Source: 官方源信息和数据/Biologic_EIS.mpr
    215: ("unknown_215", "<f4"),  # Unknown column
    217: ("THD Ewe/%", "<f4"),
    218: ("THD I/%", "<f4"),
    220: ("NSD Ewe/%", "<f4"),
    221: ("NSD I/%", "<f4"),
    223: ("NSR Ewe/%", "<f4"),
    224: ("NSR I/%", "<f4"),
    230: ("|Ewe h2|/V", "<f4"),
    231: ("|Ewe h3|/V", "<f4"),
    232: ("|Ewe h4|/V", "<f4"),
    233: ("|Ewe h5|/V", "<f4"),
    234: ("|Ewe h6|/V", "<f4"),
    235: ("|Ewe h7|/V", "<f4"),
    236: ("|I h2|/A", "<f4"),
    237: ("|I h3|/A", "<f4"),
    238: ("|I h4|/A", "<f4"),
    239: ("|I h5|/A", "<f4"),
    240: ("|I h6|/A", "<f4"),
    241: ("|I h7|/A", "<f4"),
    242: ("|E2|/V", "<f4"),
    271: ("Phase(Z1) / deg", "<f4"),
    272: ("Phase(Z2) / deg", "<f4"),
    301: ("|Z1|/Ohm", "<f4"),
    302: ("|Z2|/Ohm", "<f4"),
    331: ("Re(Z1)/Ohm", "<f4"),
    332: ("Re(Z2)/Ohm", "<f4"),
    361: ("-Im(Z1)/Ohm", "<f4"),
    362: ("-Im(Z2)/Ohm", "<f4"),
    391: ("<E1>/V", "<f4"),
    392: ("<E2>/V", "<f4"),
    422: ("Phase(Zstack)/deg", "<f4"),
    423: ("|Zstack|/Ohm", "<f4"),
    424: ("Re(Zstack)/Ohm", "<f4"),
    425: ("-Im(Zstack)/Ohm", "<f4"),
    426: ("<Estack>/V", "<f4"),
    430: ("Phase(Zwe-ce)/deg", "<f4"),
    431: ("|Zwe-ce|/Ohm", "<f4"),
    432: ("Re(Zwe-ce)/Ohm", "<f4"),
    433: ("-Im(Zwe-ce)/Ohm", "<f4"),
    434: ("(Q-Qo)/C", "<f4"),
    435: ("dQ/C", "<f4"),
    438: ("step time/s", "<f8"),
    441: ("<Ecv>/V", "<f4"),
    462: ("Temperature/°C", "<f4"),
    467: ("Q charge/discharge/mA.h", "<f8"),
    468: ("half cycle", "<u4"),
    469: ("z cycle", "<u4"),
    471: ("<Ece>/V", "<f4"),
    473: ("THD Ewe/%", "<f4"),
    474: ("THD I/%", "<f4"),
    476: ("NSD Ewe/%", "<f4"),
    477: ("NSD I/%", "<f4"),
    479: ("NSR Ewe/%", "<f4"),
    480: ("NSR I/%", "<f4"),
    486: ("|Ewe h2|/V", "<f4"),
    487: ("|Ewe h3|/V", "<f4"),
    488: ("|Ewe h4|/V", "<f4"),
    489: ("|Ewe h5|/V", "<f4"),
    490: ("|Ewe h6|/V", "<f4"),
    491: ("|Ewe h7|/V", "<f4"),
    492: ("|I h2|/A", "<f4"),
    493: ("|I h3|/A", "<f4"),
    494: ("|I h4|/A", "<f4"),
    495: ("|I h5|/A", "<f4"),
    496: ("|I h6|/A", "<f4"),
    497: ("|I h7|/A", "<f4"),
    498: ("Q charge/mA.h", "<f8"),
    499: ("Q discharge/mA.h", "<f8"),
    500: ("step time/s", "<f8"),
    501: ("Efficiency/%", "<f8"),
    502: ("Capacity/mA.h", "<f8"),
    505: ("Rdc/Ohm", "<f4"),
    509: ("Acir/Dcir Control", "<u1"),
}

# These column IDs define flags which are all stored packed in a single byte
# The values in the map are (name, bitmask, dtype)
VMPdata_colID_flag_map = {
    1: ("mode", 0x03, np.uint8),
    2: ("ox/red", 0x04, np.bool_),
    3: ("error", 0x08, np.bool_),
    21: ("control changes", 0x10, np.bool_),
    31: ("Ns changes", 0x20, np.bool_),
    65: ("counter inc.", 0x80, np.bool_),
}


def parse_biologic_date(date_text: bytes | str) -> date:
    """Parse date from Bio-Logic files."""
    date_formats = ["%m/%d/%y", "%m-%d-%y", "%m.%d.%y"]
    date_string = date_text.decode("ascii") if isinstance(date_text, bytes) else date_text
    for date_format in date_formats:
        try:
            tm = time.strptime(date_string, date_format)
            return date(tm.tm_year, tm.tm_mon, tm.tm_mday)
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp {date_string!r}")


def vmpdata_dtype_from_col_ids(col_ids, error_on_unknown_column: bool = True):
    """Get a numpy record type from a list of column ID numbers.

    The binary layout of the data in the MPR file is described by the sequence
    of column ID numbers in the file header. This function converts that
    sequence into a list that can be used with numpy dtype load data from the
    file with np.frombuffer().

    Some column IDs refer to small values which are packed into a single byte.
    The second return value is a dict describing the bit masks with which to
    extract these columns from the flags byte.

    If error_on_unknown_column is True, an error will be raised if an unknown
    column ID is encountered. If it is False, a warning will be emited and attempts
    will be made to read the column with a few different dtypes.


    """
    type_list = []
    field_name_counts = defaultdict(int)
    flags_dict = OrderedDict()
    for col_id in col_ids:
        if col_id in VMPdata_colID_flag_map:
            # Some column IDs represent boolean flags or small integers
            # These are all packed into a single 'flags' byte whose position
            # in the overall record is determined by the position of the first
            # column ID of flag type. If there are several flags present,
            # there is still only one 'flags' int
            if "flags" not in field_name_counts:
                type_list.append(("flags", "u1"))
                field_name_counts["flags"] = 1
            flag_name, flag_mask, flag_type = VMPdata_colID_flag_map[col_id]
            # TODO what happens if a flag colID has already been seen
            # i.e. if flag_name is already present in flags_dict?
            # Does it create a second 'flags' byte in the record?
            flags_dict[flag_name] = (np.uint8(flag_mask), flag_type)
        elif col_id in VMPdata_colID_dtype_map:
            field_name, field_type = VMPdata_colID_dtype_map[col_id]
            field_name_counts[field_name] += 1
            count = field_name_counts[field_name]
            unique_field_name = "%s %d" % (field_name, count) if count > 1 else field_name
            type_list.append((unique_field_name, field_type))
        else:
            if error_on_unknown_column:
                raise ValueError(f"Unknown column ID {col_id}")
            type_list.append(("unknown_colID_%d" % col_id, UNKNOWN_COLUMN_TYPE_HIERARCHY[0]))

    return type_list, flags_dict


def read_vmp_modules(fileobj, read_module_data=True):
    """Reads in module headers in the VMPmodule_hdr format. Yields a dict with
    the headers and offset for each module.

    N.B. the offset yielded is the offset to the start of the data i.e. after
    the end of the header. The data runs from (offset) to (offset+length)"""
    while True:
        module_magic = fileobj.read(len(b"MODULE"))
        if len(module_magic) == 0:  # end of file
            break
        elif module_magic != b"MODULE":
            raise ValueError("Found %r, expecting start of new VMP MODULE" % module_magic)
        vmp_module_hdr = VMPmodule_hdr_v1

        # Reading headers binary information
        hdr_bytes = fileobj.read(vmp_module_hdr.itemsize)
        if len(hdr_bytes) < vmp_module_hdr.itemsize:
            raise IOError("Unexpected end of file while reading module header")

        # Checking if EC-Lab version is >= 11.50
        if hdr_bytes[35:39] == b"\xff\xff\xff\xff":
            vmp_module_hdr = VMPmodule_hdr_v2
            hdr_bytes += fileobj.read(VMPmodule_hdr_v2.itemsize - VMPmodule_hdr_v1.itemsize)

        hdr = np.frombuffer(hdr_bytes, dtype=vmp_module_hdr, count=1)
        hdr_dict = {n: hdr[n][0] for n in vmp_module_hdr.names or []}
        hdr_dict["offset"] = fileobj.tell()
        if read_module_data:
            hdr_dict["data"] = fileobj.read(hdr_dict["length"])
            if len(hdr_dict["data"]) != hdr_dict["length"]:
                raise IOError(
                    """Unexpected end of file while reading data
                    current module: %s
                    length read: %d
                    length expected: %d"""
                    % (
                        hdr_dict["longname"],
                        len(hdr_dict["data"]),
                        hdr_dict["length"],
                    )
                )
            yield hdr_dict
        else:
            yield hdr_dict
            fileobj.seek(hdr_dict["offset"] + hdr_dict["length"], SEEK_SET)


def loop_from_file(file: str | Path, encoding: str = "latin1") -> np.ndarray:
    """Read loop index file."""
    with open(file, "r", encoding=encoding) as f:
        line = f.readline().strip()
        if line != LOOP_MAGIC:
            raise ValueError("Invalid magic for LOOP.txt file")
        return np.array([int(line) for line in f], dtype="u4")


def timestamp_from_file(file: str | Path, encoding: str = "latin1") -> datetime:
    """Read timestamp from MPL file."""
    with open(file, "r", encoding=encoding) as f:
        line = f.readline().strip()
        if line not in LOG_MAGIC:
            raise ValueError("Invalid magic for .mpl file")
        log = f.read()
    start = tuple(map(int, re.findall(r"Acquisition started on : (\d+)\/(\d+)\/(\d+) (\d+):(\d+):(\d+)\.(\d+)", "".join(log))[0]))
    return datetime(int(start[2]), start[0], start[1], start[3], start[4], start[5], int(start[6]) * 1000)


LOG_MAGIC = "EC-Lab LOG FILEBT-Lab LOG FILE"
LOOP_MAGIC = "VMP EXPERIMENT LOOP INDEXES"


# BiologicDataReader
# ======================================================================


class BiologicMPTReader(HasTraits):
    """Reader for BioLogic MPT files."""

    filepath = Unicode()
    active_material_mass = Unicode(allow_none=True)
    sample_name = Unicode(None, allow_none=True)
    start_time = Unicode(None, allow_none=True)
    instrument = Unicode("BioLogic", allow_none=True)
    operator = Unicode(None, allow_none=True)
    wave_number = Unicode(None, allow_none=True)
    technique = List(Unicode(), default_value=["echem"])

    def __init__(self, filepath: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self) -> tuple[RawData, RawDataInfo]:
        """Load BioLogic MPT file(s) and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath not set")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.is_file():
            return self._load_single_file(path)
        elif path.is_dir():
            return self._load_directory(path)
        else:
            raise ValueError(f"Path is neither a file nor a directory: {path}")

    def _load_single_file(self, path: Path) -> tuple[RawData, RawDataInfo]:
        """Internal method to load a single BioLogic MPT file."""
        # Read raw data and metadata
        mpt_array, comments = mpt_file(path)

        # Parse and clean metadata
        file_info = self._parse_mpt_metadata(list(comments))
        metadata = {
            "file_info": file_info,
            "file_type": "MPT",
            "file_path": str(path),
        }
        cleaned_metadata = self._clean_metadata(metadata)

        # Determine mass: priority to self.active_material_mass, then metadata
        test_info = cleaned_metadata.get("test_information", {})
        mass = self.active_material_mass or test_info.get("Mass of active material") or cleaned_metadata.get("channel_process_info", {}).get("Characteristic mass")

        # Determine specific technique
        tech_list = self._detect_techniques(cleaned_metadata, mpt_array)

        # Clean data (returns xarray.Dataset)
        ds = self._clean_data(mpt_array, metadata=cleaned_metadata, active_material_mass=mass)

        # Create RawData and RawDataInfo
        raw_data = RawData(data=ds)
        raw_info = RawDataInfo(
            sample_name=self.sample_name or str(test_info.get("name", "Unknown")),
            start_time=self.start_time or test_info.get("Acquisition started on"),
            operator=self.operator or test_info.get("Operator"),
            technique=self.technique if self.technique != ["echem"] else tech_list,
            instrument=self.instrument,
            active_material_mass=str(mass) if mass is not None else None,
            wave_number=self.wave_number,
            others=cleaned_metadata,
        )

        return raw_data, raw_info

    def _load_directory(self, path: Path) -> tuple[RawData, RawDataInfo]:
        """Load all BioLogic MPT files in a directory and its subdirectories into a DataTree."""
        mpt_files = sorted(path.rglob("*.mpt"))
        if not mpt_files:
            raise FileNotFoundError(f"No .mpt files found in {path}")

        # Create a root DataTree
        tree = xr.DataTree(name=path.name)
        infos = []

        for f in mpt_files:
            try:
                raw_data, raw_info = self._load_single_file(f)
                ds = raw_data.data

                # Sanitize variable names for DataTree (replace / with _)
                # DataTree does not allow / in variable names as it's a path separator
                rename_dict = {str(var): str(var).replace("/", "_") for var in ds.data_vars if "/" in str(var)}
                if rename_dict:
                    ds = ds.rename(rename_dict)

                # Determine relative path for the tree
                rel_path = f.relative_to(path)

                # Build the path string for DataTree (using / as separator)
                # We use the filename (without extension) as the leaf node name
                node_path = "/".join(rel_path.with_suffix("").parts)

                # Add to tree
                tree[node_path] = ds

                # Store metadata in node attributes
                tree[node_path].attrs.update(raw_info.to_dict())

                infos.append(raw_info)
            except Exception as e:
                logger.warning("Failed to load %s: %s", f, e)

        if not tree.children and not tree.has_data:
            raise RuntimeError(f"Failed to load any .mpt files from {path}")

        # Merge RawDataInfo for the root
        merged_info = self._merge_infos(infos, path)

        return RawData(data=tree), merged_info

    def _merge_infos(self, infos: list[RawDataInfo], root_path: Path) -> RawDataInfo:
        """Merge multiple RawDataInfo objects into one."""
        if not infos:
            return RawDataInfo()

        # Use the first one as base
        base = infos[0]

        # Collect all techniques
        all_techs = set()
        for info in infos:
            for t in info.technique:
                all_techs.add(t)

        # Create merged info
        merged_info = RawDataInfo(
            sample_name=self.sample_name or root_path.name,
            start_time=self.start_time or base.start_time,
            operator=self.operator or base.operator,
            technique=list(all_techs),
            instrument=self.instrument or base.instrument,
            active_material_mass=self.active_material_mass or base.active_material_mass,
            wave_number=self.wave_number or base.wave_number,
            others={
                "merged_files": [str(info.get("file_path")) for info in infos if info.get("file_path")],
                "n_files": len(infos),
                "structure": "DataTree",
            },
        )

        return merged_info

    def _detect_techniques(self, cleaned_metadata: dict, mpt_array: np.ndarray) -> list[str]:
        """Detect specific electrochemical techniques from metadata and data."""
        tech_str = cleaned_metadata.get("file_info", {}).get("technique", "")
        names = mpt_array.dtype.names or []
        is_peis = "Electrochemical Impedance" in tech_str or "PEIS" in tech_str
        is_gpcl = "Galvanostatic Cycling" in tech_str or "GPCL" in tech_str
        is_ocv = "Open Circuit Voltage" in tech_str or "OCV" in tech_str
        if not is_peis and "freq/Hz" in names:
            is_peis = True

        tech_list = list(self.technique)
        if is_peis:
            tech_list.append("peis")
        if is_gpcl:
            tech_list.append("gpcl")
        if is_ocv:
            tech_list.append("ocv")
        return tech_list

    @staticmethod
    def _parse_mpt_metadata(comments: list[bytes | str]) -> dict[str, Any]:
        """Parse MPT file comments into structured metadata."""
        meta: dict[str, Any] = {}
        state = {"current_section": None, "in_parameters": False, "work_mode_list": []}

        for line in comments:
            text = line.decode("latin1") if isinstance(line, bytes) else line
            text = text.rstrip("\r\n")
            if not text.strip():
                if state["in_parameters"]:
                    state["in_parameters"] = False
                continue

            BiologicMPTReader._handle_mpt_line(text, meta, state)

        if state["work_mode_list"]:
            meta["work_mode"] = state["work_mode_list"]
        return meta

    @staticmethod
    def _handle_mpt_line(text: str, meta: dict, state: dict):
        """Handle a single line of MPT metadata."""
        indent = len(text) - len(text.lstrip())
        content = text.strip()

        def split_kv(c: str) -> tuple[str, str] | None:
            for sep in (" : ", ":"):
                if sep in c:
                    k, v = c.split(sep, 1)
                    return k.strip(), v.strip()
            return None

        def add_val(d: dict, key: str, val: Any) -> None:
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
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata to keep only essential fields from BioLogic files."""
        cleaned: dict[str, Any] = {}
        file_info = metadata.get("file_info", {})

        # Essential keys
        test_keys = [
            "technique",
            "Electrode material",
            "Electrolyte",
            "Mass of active material",
            "Reference electrode",
            "Acquisition started on",
            "Operator",
        ]
        test_info = {k: file_info[k] for k in test_keys if k in file_info}

        if "Saved on" in file_info:
            saved = file_info["Saved on"]
            if isinstance(saved, dict):
                if "File" in saved:
                    test_info["name"] = saved["File"]
                if "Directory" in saved:
                    test_info["file_path"] = saved["Directory"]

        if "file_path" in file_info and "file_path" not in test_info:
            test_info["file_path"] = file_info["file_path"]

        if test_info:
            cleaned["test_information"] = test_info
            cleaned.update({
                "sample_name": test_info.get("name"),
                "start_time": test_info.get("Acquisition started on"),
                "operator": test_info.get("Operator"),
                "active_material_mass": test_info.get("Mass of active material"),
            })

        proc_keys = ["Run on channel", "Ewe Ctrl range", "Electrode surface area", "Characteristic mass"]
        proc_info = {k: file_info[k] for k in proc_keys if k in file_info}
        if proc_info:
            cleaned["channel_process_info"] = proc_info
            if "Characteristic mass" in proc_info:
                cleaned.setdefault("active_material_mass", proc_info["Characteristic mass"])

        if "work_mode" in file_info:
            wm = file_info["work_mode"]
            cleaned["work_mode"] = wm if isinstance(wm, list) else [wm]

        cleaned["technique"] = file_info.get("technique", ["echem"])
        if isinstance(cleaned["technique"], str):
            cleaned["technique"] = [cleaned["technique"]]

        return {**cleaned, **metadata}

    @staticmethod
    def _clean_data(mpt_array: np.ndarray, metadata: dict[str, Any] | None = None, active_material_mass: Any = None) -> xr.Dataset:
        """Clean data to filter specific columns and add calculated ones."""
        n_records = len(mpt_array)
        names = list(mpt_array.dtype.names or [])

        technique = metadata.get("file_info", {}).get("technique", "") if metadata else ""
        is_peis = "Electrochemical Impedance" in technique or "PEIS" in technique
        is_gpcl = "Galvanostatic Cycling" in technique or "GPCL" in technique
        is_ocv = "Open Circuit Voltage" in technique or "OCV" in technique
        if not is_peis and "freq/Hz" in names:
            is_peis = True

        # 定义不同技术的固定列名顺序
        if is_peis:
            ordered_cols = ["cycle number", "freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm", "|Z|/Ohm", "Phase(Z)/deg"]
        elif is_gpcl:
            ordered_cols = [
                "time/s",
                "systime",
                "cycle number",
                "Ewe/V",
                "Ece/V",
                "voltage/V",
                "I/mA",
                "Capacity/mA.h",
                "SpeCap_cal/mAh/g",
            ]
        elif is_ocv:
            ordered_cols = [
                "time/s",
                "systime",
                "cycle number",
                "Ewe/V",
                "Ece/V",
                "voltage/V",
            ]
        else:
            ordered_cols = list(names)

        # 1. 提取原始列
        data_vars = {col: (["record"], mpt_array[col]) for col in ordered_cols if col in names}
        coords = {"record": np.arange(1, n_records + 1)}

        # 2. 添加计算列和坐标
        if is_gpcl:
            BiologicMPTReader._add_gpcl_columns(data_vars, coords, mpt_array, names, metadata, active_material_mass)
        elif is_ocv:
            BiologicMPTReader._add_ocv_columns(data_vars, coords, mpt_array, names, metadata)

        # 3. 按照固定顺序重新构建 (仅保留存在的列)
        final_vars = {}
        for col in ordered_cols:
            if col in data_vars:
                final_vars[col] = data_vars[col]

        return xr.Dataset(final_vars, coords=coords)

    @staticmethod
    def _add_gpcl_columns(data_vars: dict, coords: dict, mpt_array: np.ndarray, names: list[str], metadata: dict | None, mass: Any):  # noqa: PLR0913, PLR0917
        """Add calculated columns for GPCL technique."""
        # voltage/V
        ewe, ece = (mpt_array[k] if k in names else None for k in ("Ewe/V", "Ece/V"))
        if ewe is not None and ece is not None:
            data_vars["voltage/V"] = (["record"], ewe - ece)
        elif ewe is not None:
            data_vars["voltage/V"] = (["record"], ewe)
        elif ece is not None:
            data_vars["voltage/V"] = (["record"], -ece)

        # systime and time_s
        try:
            acq_start = metadata.get("file_info", {}).get("Acquisition started on", "") if metadata else ""
            if acq_start and "time/s" in names:
                systimes = _calculate_systime(acq_start, mpt_array["time/s"])
                coords["systime"] = (["record"], systimes)
                rel_times = systimes - systimes[0]
                coords["time_s"] = (["record"], rel_times)
        except Exception as e:
            logger.debug("Failed to calculate systime: %s", e)

        # SpeCap_cal/mAh/g
        if mass and "Capacity/mA.h" in names:
            try:
                m_str = str(mass).lower()
                m_val = float("".join(c for c in m_str if c.isdigit() or c == "."))
                m_g = m_val * (0.001 if "mg" in m_str else 1.0)
                if m_g > 0:
                    data_vars["SpeCap_cal/mAh/g"] = (["record"], mpt_array["Capacity/mA.h"] / m_g)
            except (ValueError, TypeError, ZeroDivisionError) as e:
                logger.debug("Failed to calculate SpeCap_cal/mAh/g: %s", e)

    @staticmethod
    def _add_ocv_columns(data_vars: dict, coords: dict, mpt_array: np.ndarray, names: list[str], metadata: dict | None):
        """Add calculated columns for OCV technique."""
        n_records = len(mpt_array)

        # Ensure Ewe/V and Ece/V exist
        ewe = mpt_array["Ewe/V"] if "Ewe/V" in names else np.zeros(n_records)
        ece = mpt_array["Ece/V"] if "Ece/V" in names else np.zeros(n_records)

        data_vars["Ewe/V"] = (["record"], ewe)
        data_vars["Ece/V"] = (["record"], ece)

        # voltage/V
        data_vars["voltage/V"] = (["record"], ewe - ece)

        # systime and time_s
        try:
            acq_start = metadata.get("file_info", {}).get("Acquisition started on", "") if metadata else ""
            if acq_start and "time/s" in names:
                systimes = _calculate_systime(acq_start, mpt_array["time/s"])
                coords["systime"] = (["record"], systimes)
                rel_times = systimes - systimes[0]
                coords["time_s"] = (["record"], rel_times)
        except Exception as e:
            logger.debug("Failed to calculate systime: %s", e)
