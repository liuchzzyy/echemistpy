# -*- coding: utf-8 -*-
"""Bio-Logic MPT file reader with metadata extraction using traitlets.

Main classes:
- BiologicDataReader: Modern reader with traitlets support for MPT files
- BiologicMPTReader: Legacy reader interface

Based on: https://github.com/echemdata/galvani/blob/master/galvani/BioLogic.py
"""

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
from traitlets import HasTraits, Unicode

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
    technique = ["echem"]
    instrument = "BioLogic"

    def __init__(self, filepath: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self) -> tuple[RawData, RawDataInfo]:
        """Load BioLogic MPT file and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath not set")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Read raw data and metadata
        metadata, data_dict = self._read_mpt_file()

        # Clean metadata and data
        cleaned_metadata = self._clean_metadata(metadata)
        cleaned_df = self._clean_data(data_dict, metadata)

        # Convert to xarray Dataset
        ds = xr.Dataset.from_dataframe(cleaned_df)
        if "index" in ds.coords:
            ds = ds.rename({"index": "row"})
        elif "record" in ds.coords:
            ds = ds.rename({"record": "row"})
        elif "records" in ds.coords:
            ds = ds.rename({"records": "row"})

        # Create RawData and RawDataInfo
        raw_data = RawData(data=ds)

        # Extract top-level metadata from cleaned_metadata
        test_info = cleaned_metadata.get("test_information", {})
        sample_name = str(test_info.get("name", "Unknown"))
        start_time = test_info.get("Acquisition started on")
        operator = test_info.get("Operator")

        raw_info = RawDataInfo(
            sample_name=sample_name,
            start_time=start_time,
            operator=operator,
            technique=self.technique,
            instrument=self.instrument,
            others=cleaned_metadata,
        )

        return raw_data, raw_info

    def _read_mpt_file(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read metadata and data from MPT file."""
        mpt_array, comments = mpt_file(Path(self.filepath))

        metadata = {
            "file_info": self._parse_mpt_metadata(list(comments)),
            "file_type": "MPT",
            "file_path": str(self.filepath),
        }

        column_names: list[str] = list(mpt_array.dtype.names or [])

        # Ensure unique column names without normalization
        unique_col_names = []
        seen = {}
        for name in column_names:
            if name not in seen:
                unique_col_names.append(name)
                seen[name] = 1
            else:
                seen[name] += 1
                unique_col_names.append(f"{name}_{seen[name]}")

        data_dict = {unique_name: mpt_array[raw_name].tolist() for raw_name, unique_name in zip(column_names, unique_col_names, strict=True)}

        return metadata, data_dict

    @staticmethod
    def _parse_mpt_metadata(comments: list[bytes | str]) -> dict[str, Any]:
        """Parse MPT file comments into structured metadata."""
        meta: dict[str, Any] = {}
        current_section: dict[str, Any] | None = None
        in_parameters = False
        work_mode_list: list[dict[str, Any]] = []

        def split_key_value(content: str) -> tuple[str, str] | None:
            """Split key-value pair, trying ' : ' first, then ':'."""
            if " : " in content:
                key, value = content.split(" : ", 1)
            elif ":" in content:
                key, value = content.split(":", 1)
            else:
                return None
            return key.strip(), value.strip()

        def add_or_update_value(d: dict[str, Any], key: str, value: Any) -> None:
            """Add a value, converting duplicates to list structure."""
            if key not in d:
                d[key] = value
            else:
                # Convert to list on duplicate
                existing = d[key]
                if isinstance(existing, list):
                    # Already a list, append new value
                    existing.append(value)
                else:
                    # First duplicate, convert to list
                    d[key] = [existing, value]

        for line in comments:
            text_line = line
            if isinstance(text_line, bytes):
                try:
                    text_line = text_line.decode("latin1")
                except Exception as e:
                    logger.debug("Failed to decode line as latin1: %s", e)
                    continue

            text_line = text_line.rstrip("\r\n")
            if not text_line.strip():
                # Empty line may signal end of parameters section
                if in_parameters and current_section is not None:
                    in_parameters = False
                continue

            indent_level = len(text_line) - len(text_line.lstrip())
            line_content = text_line.strip()

            if indent_level > 0 and current_section is not None:
                kv = split_key_value(line_content)
                if kv:
                    key, value = kv
                    add_or_update_value(current_section, key, value)
            else:
                if "technique" not in meta and any(kw in line_content.lower() for kw in ["electrochemical", "impedance", "spectroscopy", "potentio", "galvano"]):
                    meta["technique"] = line_content
                    continue

                if line_content.startswith("Cycle Definition"):
                    in_parameters = True
                    kv = split_key_value(line_content)
                    current_section = {"cycle_definition": kv[1]} if kv else {}
                    work_mode_list.append(current_section)
                    continue

                if in_parameters and current_section is not None:
                    # Match: parameter name (possibly with units in parentheses) followed by value
                    # e.g., "E (V)               0.0000" or "Mode                Single sine"
                    match = re.match(r"(.+?)\s{2,}(.+)", text_line)
                    if match:
                        key_part, value = match.groups()
                        add_or_update_value(current_section, key_part.strip(), value.strip())
                    elif line_content:
                        add_or_update_value(current_section, line_content, "")
                # Only add to meta if we're not in parameters section
                elif not in_parameters:
                    kv = split_key_value(line_content)
                    if kv:
                        key, value = kv
                        if not value:
                            current_section = {}
                            meta[key] = current_section
                        else:
                            add_or_update_value(meta, key, value)
                            current_section = None

        if work_mode_list:
            meta["work_mode"] = work_mode_list

        return meta

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata to keep only essential fields from BioLogic files."""
        cleaned: dict[str, Any] = {}

        if "file_info" not in metadata:
            return metadata

        file_info = metadata["file_info"]

        # Use raw keys from MPT file
        test_info_keys = ["technique", "Electrode material", "Electrolyte", "Mass of active material", "Reference electrode", "Acquisition started on", "Operator"]
        test_info = {key: file_info[key] for key in test_info_keys if key in file_info}

        if "Saved on" in file_info:
            saved_on = file_info["Saved on"]
            if isinstance(saved_on, dict):
                if "File" in saved_on and isinstance(saved_on.get("File"), str):
                    test_info["name"] = saved_on["File"]
                if "Directory" in saved_on and isinstance(saved_on.get("Directory"), str):
                    test_info["file_path"] = saved_on["Directory"]

        if "file_path" in file_info and "file_path" not in test_info:
            test_info["file_path"] = file_info["file_path"]

        if test_info:
            cleaned["test_information"] = test_info

        proc_info_keys = ["Run on channel", "Ewe Ctrl range", "Electrode surface area", "Characteristic mass"]
        proc_info = {key: file_info[key] for key in proc_info_keys if key in file_info}

        if proc_info:
            cleaned["channel_process_info"] = proc_info

        if "work_mode" in file_info:
            work_mode = file_info["work_mode"]
            cleaned["work_mode"] = work_mode if isinstance(work_mode, list) else [work_mode]

        # Ensure technique is set at top level for standardizer
        cleaned["technique"] = file_info.get("technique", ["echem"])
        if isinstance(cleaned["technique"], str):
            cleaned["technique"] = [cleaned["technique"]]

        # Merge with original metadata to ensure nothing is lost
        final_meta = cleaned.copy()
        final_meta.update(metadata)
        return final_meta

    @staticmethod
    def _clean_data(data: dict[str, Any], metadata: dict[str, Any] | None = None) -> pd.DataFrame:
        """Clean data to keep only specified columns from BioLogic data."""
        # Convert dict to DataFrame
        if isinstance(data, dict):
            data_copy = {k: v for k, v in data.items() if k != "_metadata"}
            if not data_copy:
                return pd.DataFrame()
            try:
                df = pd.DataFrame(data_copy)
            except Exception:
                return pd.DataFrame()
        else:
            return data if isinstance(data, pd.DataFrame) else pd.DataFrame()

        # Detect measurement type
        is_peis = False
        is_gpcl = False

        if metadata:
            technique = metadata.get("file_info", {}).get("technique", "")
            is_peis = "Electrochemical Impedance" in technique or "PEIS" in technique
            is_gpcl = "Galvanostatic Cycling" in technique or "GPCL" in technique

        # Fallback detection based on columns (using raw keys)
        if not is_peis and "freq/Hz" in df.columns and ("Re(Z)/Ohm" in df.columns or "-Im(Z)/Ohm" in df.columns):
            is_peis = True

        if is_peis:
            peis_mapping = {
                "record": None,
                "cycle number": "cycle number",
                "freq/Hz": "freq/Hz",
                "Re(Z)/Ohm": "Re(Z)/Ohm",
                "-Im(Z)/Ohm": "-Im(Z)/Ohm",
                "|Z|/Ohm": "|Z|/Ohm",
                "Phase(Z)/deg": "Phase(Z)/deg",
            }

            result_df = pd.DataFrame()
            result_df["record"] = range(1, len(df) + 1)

            for output_col, input_col in peis_mapping.items():
                if output_col != "record" and input_col and input_col in df.columns:
                    col_data = df[input_col]
                    # For imaginary impedance, negate the values (convert -Im(Z) to Im(Z))
                    if output_col == "-Im(Z)/Ohm":
                        col_data = -col_data
                    result_df[output_col] = col_data

            return result_df if not result_df.empty else pd.DataFrame()
        elif is_gpcl:
            # GPCL (Galvanostatic Cycling with Potential Limitation) columns (using raw keys)
            gpcl_mapping = {
                "records": None,
                "systime": None,
                "time/s": "time/s",
                "cycle number": "cycle number",
                "Ewe/V": "Ewe/V",
                "Ece/V": "Ece/V",
                "I/mA": "I/mA",
                "(Q-Qo)/mA.h": "(Q-Qo)/mA.h",
                "voltage/V": None,
            }

            result_df = pd.DataFrame()
            result_df["records"] = range(1, len(df) + 1)

            for output_col, input_col in gpcl_mapping.items():
                if output_col == "records":
                    continue
                elif output_col == "systime":
                    # Calculate systime = acquisition_started_on + time/s (YYYY-MM-DDTHH:MM:SS)
                    try:
                        acq_start = metadata.get("file_info", {}).get("Acquisition started on", "") if metadata else ""
                        if acq_start and "time/s" in df.columns:
                            start_dt = datetime.strptime(acq_start, "%m/%d/%Y %H:%M:%S.%f")
                            systime_list = [datetime.fromtimestamp(start_dt.timestamp() + time_offset).strftime("%Y-%m-%dT%H:%M:%S") for time_offset in df["time/s"]]
                            result_df["systime"] = systime_list
                        else:
                            result_df["systime"] = ""
                    except Exception:
                        result_df["systime"] = ""
                elif output_col == "voltage/V":
                    # Calculate voltage/V = Ewe/V - Ece/V
                    ewe_v = df.get("Ewe/V")
                    ece_v = df.get("Ece/V")
                    if ewe_v is not None and ece_v is not None:
                        result_df["voltage/V"] = ewe_v - ece_v
                    elif ewe_v is not None:
                        result_df["voltage/V"] = ewe_v
                    elif ece_v is not None:
                        result_df["voltage/V"] = -ece_v
                    else:
                        result_df["voltage/V"] = 0
                elif input_col and input_col in df.columns:
                    result_df[output_col] = df[input_col]
                else:
                    result_df[output_col] = 0

            return result_df if not result_df.empty else pd.DataFrame()

        return df
