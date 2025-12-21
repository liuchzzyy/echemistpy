# -*- coding: utf-8 -*-
"""Bio-Logic MPT file reader with metadata extraction using traitlets.

Main classes:
- BiologicDataReader: Modern reader with traitlets support for MPT files
- BiologicMPTReader: Legacy reader interface

Based on: https://github.com/echemdata/galvani/blob/master/galvani/BioLogic.py
"""

import argparse
import json
import logging
import re
import time
from datetime import date, datetime
from collections import defaultdict, OrderedDict
from pathlib import Path
from os import SEEK_SET
from typing import Any

import numpy as np
import pandas as pd
from traitlets import HasTraits, Union, Dict, Instance, Unicode, validate

logger = logging.getLogger(__name__)

UNKNOWN_COLUMN_TYPE_HIERARCHY = ("<f8", "<f4", "<u4", "<u2", "<u1")


def fieldname_to_dtype(fieldname: str) -> tuple[str, Any]:
    """Convert column header from MPT file to (name, dtype) tuple."""
    if fieldname == "mode":
        return ("mode", np.uint8)
    if fieldname in (
        "ox/red",
        "error",
        "control changes",
        "Ns changes",
        "counter inc.",
    ):
        return (fieldname, np.bool_)
    elif fieldname in (
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
    ):
        return (fieldname, np.float64)
    elif fieldname in (
        "Q charge/discharge/mA.h",
        "step time/s",
        "Q charge/mA.h",
        "Q discharge/mA.h",
        "Temperature/°C",
        "Efficiency/%",
        "Capacity/mA.h",
    ):
        return (fieldname, np.float64)
    elif fieldname in ("cycle number", "I Range", "Ns", "half cycle", "z cycle"):
        return (fieldname, np.int_)
    elif fieldname in ("dq/mA.h", "dQ/mA.h"):
        return ("dQ/mA.h", np.float64)
    elif fieldname in ("I/mA", "<I>/mA"):
        return ("I/mA", np.float64)
    elif fieldname in ("Ewe/V", "<Ewe>/V", "Ecell/V", "<Ewe/V>"):
        return ("Ewe/V", np.float64)
    elif fieldname.endswith((
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
    )):
        return (fieldname, np.float64)
    elif fieldname.startswith("empty_column_"):
        # Handle empty columns that were renamed
        return (fieldname, np.float64)
    else:
        raise ValueError("Invalid column header: %s" % fieldname)


def comma_converter(float_text: bytes) -> float:
    """Convert text to float, handling ',' as decimal point."""
    return float(float_text.translate(bytes.maketrans(b",", b".")))


def MPTfile(file_or_path: str | Path, encoding: str = "latin1") -> tuple[np.ndarray, list[bytes]]:
    """Open .mpt files as numpy record arrays with comments."""
    if isinstance(file_or_path, (str, Path)):
        mpt_file = open(file_or_path, "rb")
        should_close = True
    else:
        mpt_file = file_or_path
        should_close = False

    try:
        magic = next(mpt_file).strip()
        if magic not in (b"EC-Lab ASCII FILE", b"BT-Lab ASCII FILE"):
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

        converter_dict: dict[int, Any] = {i: str_to_float for i in range(len(fieldnames))}
        mpt_array = np.loadtxt(mpt_file, dtype=record_type, converters=converter_dict)  # type: ignore[arg-type]

        return mpt_array, comments
    finally:
        if should_close:
            mpt_file.close()


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
    17: ("Analog IN 2/V", "<f4"),  # Probably column 18 is Analog IN 3/V, if anyone hits this error in the future  # noqa: E501
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


def parse_BioLogic_date(date_text: bytes | str) -> date:
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


def VMPdata_dtype_from_colIDs(colIDs, error_on_unknown_column: bool = True):
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
    for colID in colIDs:
        if colID in VMPdata_colID_flag_map:
            # Some column IDs represent boolean flags or small integers
            # These are all packed into a single 'flags' byte whose position
            # in the overall record is determined by the position of the first
            # column ID of flag type. If there are several flags present,
            # there is still only one 'flags' int
            if "flags" not in field_name_counts:
                type_list.append(("flags", "u1"))
                field_name_counts["flags"] = 1
            flag_name, flag_mask, flag_type = VMPdata_colID_flag_map[colID]
            # TODO what happens if a flag colID has already been seen
            # i.e. if flag_name is already present in flags_dict?
            # Does it create a second 'flags' byte in the record?
            flags_dict[flag_name] = (np.uint8(flag_mask), flag_type)
        elif colID in VMPdata_colID_dtype_map:
            field_name, field_type = VMPdata_colID_dtype_map[colID]
            field_name_counts[field_name] += 1
            count = field_name_counts[field_name]
            if count > 1:
                unique_field_name = "%s %d" % (field_name, count)
            else:
                unique_field_name = field_name
            type_list.append((unique_field_name, field_type))
        else:
            if error_on_unknown_column:
                raise ValueError(f"Unknown column ID {colID}")
            type_list.append(("unknown_colID_%d" % colID, UNKNOWN_COLUMN_TYPE_HIERARCHY[0]))

    return type_list, flags_dict


def read_VMP_modules(fileobj, read_module_data=True):
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
        VMPmodule_hdr = VMPmodule_hdr_v1

        # Reading headers binary information
        hdr_bytes = fileobj.read(VMPmodule_hdr.itemsize)
        if len(hdr_bytes) < VMPmodule_hdr.itemsize:
            raise IOError("Unexpected end of file while reading module header")

        # Checking if EC-Lab version is >= 11.50
        if hdr_bytes[35:39] == b"\xff\xff\xff\xff":
            VMPmodule_hdr = VMPmodule_hdr_v2
            hdr_bytes += fileobj.read(VMPmodule_hdr_v2.itemsize - VMPmodule_hdr_v1.itemsize)

        hdr = np.frombuffer(hdr_bytes, dtype=VMPmodule_hdr, count=1)
        hdr_dict = dict(((n, hdr[n][0]) for n in VMPmodule_hdr.names or []))
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


class BiologicDataReader(HasTraits):
    """Reader for BioLogic MPT files with traitlets support and metadata extraction."""

    # Traitlets properties
    filepath = Union([Instance(Path), Unicode()], allow_none=True, help="Path to BioLogic MPT file or folder").tag(config=True)

    metadata = Dict(help="Extracted metadata from BioLogic file")
    data = Dict(help="Measurement data from BioLogic file")

    def __init__(self, filepath: Path | str | None = None, **kwargs):
        """Initialize reader with optional filepath using traitlets."""
        super().__init__(**kwargs)
        if filepath is not None:
            self.filepath = filepath  # Validator will handle conversion

    @validate("filepath")
    def _validate_filepath(self, proposal):
        """Validate filepath and convert to Path if needed."""
        value = proposal["value"]
        if value is None or isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        raise TypeError(f"filepath must be Path or str, got {type(value)}")

    def _read_biologic_file(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read metadata and data from BioLogic MPT file."""
        if not self.filepath:
            raise ValueError("filepath not set")

        suffix = self.filepath.suffix.lower()
        if suffix == ".mpt":
            return self._read_mpt_file()
        else:
            raise ValueError(f"Only .mpt files are supported. Received: {suffix}")

    def _read_mpt_file(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read metadata and data from MPT file."""
        mpt_array, comments = MPTfile(self.filepath)

        metadata = {
            "file_info": self._parse_mpt_metadata(list(comments)),
            "file_type": "MPT",
            "file_path": str(self.filepath),
        }

        column_names: list[str] = list(mpt_array.dtype.names or [])
        _, unique_col_names = self._get_unique_column_names(column_names)

        data_dict = {unique_name: mpt_array[raw_name].tolist() for raw_name, unique_name in zip(column_names, unique_col_names)}
        data_dict["_metadata"] = {
            "file_type": "MPT",
            "num_rows": len(mpt_array),
            "num_columns": len(column_names),
            "columns": unique_col_names,
        }

        return self._normalize_dict_keys(metadata), data_dict

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
            if isinstance(line, bytes):
                try:
                    line = line.decode("latin1")
                except Exception:
                    continue

            line = line.rstrip("\r\n")
            if not line.strip():
                # Empty line may signal end of parameters section
                if in_parameters and current_section is not None:
                    in_parameters = False
                continue

            indent_level = len(line) - len(line.lstrip())
            line_content = line.strip()

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
                    match = re.match(r"(.+?)\s{2,}(.+)", line)
                    if match:
                        key_part, value = match.groups()
                        # Normalize the key by removing units in parentheses
                        key_normalized = BiologicDataReader._normalize_key(key_part.strip())
                        add_or_update_value(current_section, key_normalized, value.strip())
                    elif line_content:
                        add_or_update_value(current_section, BiologicDataReader._normalize_key(line_content), "")
                else:
                    # Only add to meta if we're not in parameters section
                    if not in_parameters:
                        kv = split_key_value(line_content)
                        if kv:
                            key, value = kv
                            if value == "":
                                current_section = {}
                                meta[key] = current_section
                            else:
                                add_or_update_value(meta, key, value)
                                current_section = None

        if work_mode_list:
            meta["work_mode"] = work_mode_list

        return meta

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize key: lowercase, handle special chars."""
        key = str(key).lower()
        # Remove brackets and pipes, replace special chars with underscores
        for char, replacement in [("<", ""), (">", ""), ("|", ""), ("/", "_"), (" ", "_"), ("-", "_"), ("(", ""), (")", "")]:
            key = key.replace(char, replacement)
        # Collapse multiple underscores
        return re.sub(r"_+", "_", key).strip("_")

    @staticmethod
    def _get_unique_column_names(raw_column_names: list[str]) -> tuple[list[str], list[str]]:
        """Convert raw column names to unique normalized names."""
        normalized_names = [BiologicDataReader._normalize_key(name) for name in raw_column_names]
        name_counts: dict[str, list[int]] = defaultdict(list)

        for i, norm_name in enumerate(normalized_names):
            name_counts[norm_name].append(i)

        unique_names = []
        for i, norm_name in enumerate(normalized_names):
            if len(name_counts[norm_name]) == 1:
                unique_names.append(norm_name)
            else:
                occurrence_num = name_counts[norm_name].index(i)
                unique_names.append(f"{norm_name}_{chr(ord('a') + occurrence_num)}")

        final_unique_names = []
        seen = set()
        for original_name in unique_names:
            if original_name not in seen:
                final_unique_names.append(original_name)
                seen.add(original_name)
            else:
                counter = 2
                while f"{original_name}_{counter}" in seen:
                    counter += 1
                new_name = f"{original_name}_{counter}"
                final_unique_names.append(new_name)
                seen.add(new_name)

        return normalized_names, final_unique_names

    @staticmethod
    def _normalize_dict_keys(data: dict) -> dict:
        """Recursively normalize all dictionary keys to lowercase and replace special chars."""
        if not isinstance(data, dict):
            return data

        normalized: dict[str, Any] = {}
        for key, value in data.items():
            # Skip empty dicts
            if isinstance(value, dict) and not value:
                continue

            new_key = BiologicDataReader._normalize_key(key)

            # Recursively process nested structures
            if isinstance(value, dict):
                normalized[new_key] = BiologicDataReader._normalize_dict_keys(value)
            elif isinstance(value, list):
                normalized[new_key] = [BiologicDataReader._normalize_dict_keys(item) if isinstance(item, dict) else item for item in value]
            else:
                normalized[new_key] = value

        return normalized

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata to keep only essential fields from BioLogic files."""
        cleaned: dict[str, Any] = {}

        if "file_info" not in metadata:
            return cleaned

        file_info = metadata["file_info"]

        test_info_keys = ["technique", "electrode_material", "electrolyte", "mass_of_active_material", "reference_electrode", "acquisition_started_on"]
        test_info = {key: file_info[key] for key in test_info_keys if key in file_info}

        if "saved_on" in file_info:
            saved_on = file_info["saved_on"]
            if isinstance(saved_on, dict):
                if "file" in saved_on and isinstance(saved_on.get("file"), str):
                    test_info["name"] = saved_on["file"]
                if "directory" in saved_on and isinstance(saved_on.get("directory"), str):
                    test_info["file_path"] = saved_on["directory"]

        if "file_path" in file_info and "file_path" not in test_info:
            test_info["file_path"] = file_info["file_path"]

        if test_info:
            cleaned["test_information"] = test_info

        proc_info_keys = ["run_on_channel", "ewe_ctrl_range", "electrode_surface_area", "characteristic_mass"]
        proc_info = {key: file_info[key] for key in proc_info_keys if key in file_info}

        if proc_info:
            cleaned["channel_process_info"] = proc_info

        if "work_mode" in file_info:
            work_mode = file_info["work_mode"]
            cleaned["work_mode"] = work_mode if isinstance(work_mode, list) else [work_mode]

        return cleaned

    @staticmethod
    def _clean_data(data: dict[str, Any], metadata: dict[str, Any] | None = None) -> pd.DataFrame:
        """Clean data to keep only specified columns from BioLogic measurements."""
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

        # Fallback detection based on columns
        if not is_peis and "freq_hz" in df.columns and ("rez_ohm" in df.columns or "re_z_ohm" in df.columns):
            is_peis = True

        if is_peis:
            peis_mapping = {
                "record": None,
                "cycle_number": "cycle_number",
                "freq_hz": "freq_hz",
                "re_z_ohm": "rez_ohm",
                "im_z_ohm": "imz_ohm",
                "z_ohm": "z_ohm",
                "phase_z_deg": "phasez_deg",
            }

            result_df = pd.DataFrame()
            result_df["record"] = range(1, len(df) + 1)

            for output_col, input_col in peis_mapping.items():
                if output_col != "record" and input_col and input_col in df.columns:
                    col_data = df[input_col]
                    # For imaginary impedance, negate the values (convert -Im(Z) to Im(Z))
                    if output_col == "im_z_ohm":
                        col_data = -col_data
                    result_df[output_col] = col_data

            return result_df if not result_df.empty else pd.DataFrame()
        elif is_gpcl:
            # GPCL (Galvanostatic Cycling with Potential Limitation) columns
            gpcl_mapping = {
                "records": None,
                "systime": None,
                "time_s": "time_s",
                "cycle_number": "cycle_number",
                "ewe_v": "ewe_v",
                "ece_v": "ece_v",
                "i_ma": "i_ma",
                "capacity_ma.h": "capacity_ma.h",
                "voltage_v": None,
            }

            result_df = pd.DataFrame()
            result_df["records"] = range(1, len(df) + 1)

            for output_col, input_col in gpcl_mapping.items():
                if output_col == "records":
                    continue
                elif output_col == "systime":
                    # Calculate systime = acquisition_started_on + time_s (YYYY-MM-DDTHH:MM:SS)
                    try:
                        acq_start = metadata.get("file_info", {}).get("acquisition_started_on", "") if metadata else ""
                        if acq_start and "time_s" in df.columns:
                            start_dt = datetime.strptime(acq_start, "%m/%d/%Y %H:%M:%S.%f")
                            systime_list = [datetime.fromtimestamp(start_dt.timestamp() + time_offset).strftime("%Y-%m-%dT%H:%M:%S") for time_offset in df["time_s"]]
                            result_df["systime"] = systime_list
                        else:
                            result_df["systime"] = ""
                    except Exception:
                        result_df["systime"] = ""
                elif output_col == "voltage_v":
                    # Calculate voltage_v = ewe_v - ece_v
                    ewe_v = df.get("ewe_v")
                    ece_v = df.get("ece_v")
                    if ewe_v is not None and ece_v is not None:
                        result_df["voltage_v"] = ewe_v - ece_v
                    elif ewe_v is not None:
                        result_df["voltage_v"] = ewe_v
                    elif ece_v is not None:
                        result_df["voltage_v"] = -ece_v
                    else:
                        result_df["voltage_v"] = 0
                elif input_col and input_col in df.columns:
                    result_df[output_col] = df[input_col]
                else:
                    result_df[output_col] = 0

            return result_df if not result_df.empty else pd.DataFrame()

        return df

    @staticmethod
    def _save_data(data: dict[str, Any] | list[Any] | pd.DataFrame, output_path: Path, data_type: str, cleaned: bool = False) -> None:
        """Save metadata as JSON or data as CSV."""
        try:
            if cleaned:
                output_path = output_path.with_stem(output_path.stem + "_cleaned")
            if data_type == "metadata":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                if isinstance(data, pd.DataFrame):
                    df = data
                elif isinstance(data, dict):
                    data_copy = {k: v for k, v in data.items() if k != "_metadata"}
                    if not data_copy:
                        return
                    df = pd.DataFrame(data_copy)
                else:
                    return
                df.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        except IOError as e:
            logger.error(f"Failed to save {data_type} to {output_path}: {e}")

    @staticmethod
    def _read_all_biologic(folder_path: Path | str | None = None) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        """Read all BioLogic MPT files from folder."""
        if folder_path is None:
            return {}

        folder_path = Path(folder_path) if isinstance(folder_path, str) else folder_path
        reader = BiologicDataReader()
        all_data = {}

        for bio_file in folder_path.glob("*.mpt"):
            try:
                reader.filepath = bio_file
                metadata, data = reader._read_biologic_file()
                all_data[bio_file.name] = (metadata, data)
            except Exception as e:
                logger.error(f"Error processing {bio_file.name}: {e}", exc_info=True)

        return all_data

    @staticmethod
    def _ensure_path(path: Path | str) -> Path:
        """Convert string to Path if necessary."""
        return Path(path) if isinstance(path, str) else path

    @staticmethod
    def load(filepath: Path | str) -> tuple[dict[str, Any], dict[str, Any]] | dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        """Load BioLogic file or folder and return data."""
        filepath = BiologicDataReader._ensure_path(filepath)
        if filepath.is_dir():
            return BiologicDataReader._read_all_biologic(filepath)
        if filepath.is_file():
            reader = BiologicDataReader(filepath)
            return reader._read_biologic_file()
        raise FileNotFoundError(f"Path not found: {filepath}")

    @staticmethod
    def _process_and_save(metadata: dict[str, Any], data: dict[str, Any], metadata_file: Path, data_file: Path, save_original: bool = False, save_cleaned: bool = True) -> None:
        """Process and save metadata and data files."""
        if save_original:
            BiologicDataReader._save_data(metadata, metadata_file, "metadata", cleaned=False)
            BiologicDataReader._save_data(data, data_file, "data", cleaned=False)

        if save_cleaned:
            cleaned_metadata = BiologicDataReader._clean_metadata(metadata)
            cleaned_data = BiologicDataReader._clean_data(data, metadata)
            BiologicDataReader._save_data(cleaned_metadata, metadata_file, "metadata", cleaned=True)
            BiologicDataReader._save_data(cleaned_data, data_file, "data", cleaned=True)

    @staticmethod
    def save(input_filepath: Path | str, output_dir: Path | str, save_cleaned: bool = True, save_original: bool = False) -> None:
        """Load from input and save metadata (JSON) and data (CSV).

        Parameters
        ----------
        input_filepath : Path or str
            Input BioLogic file or folder path
        output_dir : Path or str
            Output directory
        save_cleaned : bool
            If True, save cleaned versions (default: True)
        save_original : bool
            If True, also save original (non-cleaned) versions (default: False)
        """
        input_filepath = BiologicDataReader._ensure_path(input_filepath)
        output_dir = BiologicDataReader._ensure_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = BiologicDataReader.load(input_filepath)

        if input_filepath.is_file():
            if isinstance(result, tuple):
                metadata, data = result
                base_name = input_filepath.stem
                metadata_file = output_dir / f"{base_name}_metadata.json"
                data_file = output_dir / f"{base_name}_data.json"
                BiologicDataReader._process_and_save(metadata, data, metadata_file, data_file, save_original, save_cleaned)
        else:
            if isinstance(result, dict):
                for filename, (metadata, data) in result.items():
                    file_path = Path(filename)
                    base_name = f"{file_path.stem}_{file_path.suffix.lstrip('.')}"
                    metadata_file = output_dir / f"{base_name}_metadata.json"
                    data_file = output_dir / f"{base_name}_data.json"
                    BiologicDataReader._process_and_save(metadata, data, metadata_file, data_file, save_original, save_cleaned)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="BiologicMPTReader for BioLogic files", prog="BiologicMPTReader.py")
    parser.add_argument("path", type=str, help="Path to BioLogic file or folder")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory (default: output)")
    parser.add_argument("--no-clean", action="store_true", help="Also save original (non-cleaned) data in addition to cleaned data")

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path not found: {path}")
        exit(1)

    # save_cleaned=True, save_original=args.no_clean
    # When --no-clean is passed, save_original=True to also save original data
    BiologicDataReader.save(path, args.output, save_cleaned=True, save_original=args.no_clean)
