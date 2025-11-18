"""Generic file loading helpers to keep notebooks lean.

This module provides comprehensive data loading capabilities for various file formats
commonly used in scientific measurements. It serves as the primary entry point for
reading raw data files before they are processed by the organization module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import xarray as xr

Loader = Callable[..., xr.Dataset]


def _structured_array_to_dataset(array: np.ndarray) -> xr.Dataset:
    if array.size == 0:
        return xr.Dataset()
    if array.ndim == 0:
        array = array.reshape(1)
    if array.dtype.names is None:
        raise ValueError("Tabular text files must include a header row.")
    row_dim = np.arange(array.shape[0])
    data_vars = {name: ("row", array[name]) for name in array.dtype.names}
    return xr.Dataset(data_vars=data_vars, coords={"row": row_dim})


def _load_delimited(path: Path, *, delimiter: str, **kwargs: Any) -> xr.Dataset:
    array = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=None, encoding=None, **kwargs)
    return _structured_array_to_dataset(array)


def _load_json_table(path: Path, **_: Any) -> xr.Dataset:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        records: Iterable[Mapping[str, Any]] = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("JSON tables must be a dict or list of dicts.")
    records = list(records)
    if not records:
        return xr.Dataset()
    keys = sorted({key for record in records for key in record})
    data_vars = {key: ("row", np.asarray([record.get(key) for record in records], dtype=object)) for key in keys}
    return xr.Dataset(data_vars=data_vars, coords={"row": np.arange(len(records))})


_LOADER_MAP: Dict[str, Loader] = {
    "csv": lambda path, **kwargs: _load_delimited(path, delimiter=",", **kwargs),
    "tsv": lambda path, **kwargs: _load_delimited(path, delimiter="\t", **kwargs),
    "json": _load_json_table,
    "nc": lambda path, **kwargs: xr.open_dataset(path, **kwargs),
    "nc4": lambda path, **kwargs: xr.open_dataset(path, **kwargs),
    "netcdf": lambda path, **kwargs: xr.open_dataset(path, **kwargs),
}

# Alias for backwards compatibility
_LOADERS = _LOADER_MAP


def load_table(path: str | Path, *, fmt: Optional[str] = None, storage_options: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> xr.Dataset:
    """Load a tabular file into an :class:`xarray.Dataset`.

    Examples
    --------
    The helper understands multiple formats; JSON tables make it easy to craft
    doctest-sized fixtures.

    >>> import json, tempfile
    >>> from pathlib import Path
    >>> payload = [
    ...     {"voltage": 3.1, "current": 0.5},
    ...     {"voltage": 3.4, "current": 0.4},
    ... ]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     table_path = Path(tmp) / "trace.json"
    ...     _ = table_path.write_text(json.dumps(payload))
    ...     dataset = load_table(table_path)
    >>> dataset["voltage"].values.tolist()
    [3.1, 3.4]
    >>> dataset.row.size
    2
    """

    del storage_options  # not used yet but kept for API stability

    path = Path(path)
    extension = (fmt or path.suffix.lstrip(".")).lower()
    try:
        loader = _LOADER_MAP[extension]
    except KeyError as exc:
        raise ValueError(f"Unsupported file extension '{extension}'.") from exc
    return loader(path, **kwargs)


def register_loader(extension: str, loader: Loader) -> None:
    """Allow users to plug-in custom loaders (e.g. for proprietary formats)."""

    _LOADER_MAP[extension.lower()] = loader


def _load_excel(path: Path, **kwargs: Any) -> xr.Dataset:
    """Load Excel files using pandas backend."""
    try:
        import pandas as pd  # noqa: E402
    except ImportError as exc:
        raise ImportError("pandas is required to read Excel files") from exc

    # Read Excel file
    df = pd.read_excel(path, **kwargs)

    # Convert to xarray Dataset
    data_vars = {}
    for col in df.columns:
        data_vars[str(col)] = ("row", df[col].values)

    return xr.Dataset(data_vars=data_vars, coords={"row": np.arange(len(df))})


def _load_hdf5(path: Path, **kwargs: Any) -> xr.Dataset:
    """Load HDF5 files using xarray backend."""
    try:
        return xr.open_dataset(path, engine="h5netcdf", **kwargs)
    except Exception:
        # Fallback to netcdf4 engine
        return xr.open_dataset(path, engine="netcdf4", **kwargs)


def _auto_detect_delimiter(path: Path) -> str:
    """Auto-detect delimiter for text files by sampling first few lines."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(1024)  # Read first 1KB

    # Count common delimiters
    delimiters = {",": sample.count(","), "\t": sample.count("\t"), ";": sample.count(";"), "|": sample.count("|")}

    # Return delimiter with highest count
    return max(delimiters.keys(), key=lambda k: delimiters[k])


def _load_auto_delimited(path: Path, **kwargs: Any) -> xr.Dataset:
    """Auto-detect delimiter and load delimited file."""
    delimiter = _auto_detect_delimiter(path)
    return _load_delimited(path, delimiter=delimiter, **kwargs)


def load_data_file(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Universal data file loader with automatic format detection.

    This function attempts to load various scientific data formats by:
    1. First trying the specific loader based on file extension
    2. If that fails, attempting common scientific file formats
    3. Finally falling back to auto-detection for text files

    Args:
        path: Path to the data file
        **kwargs: Additional arguments passed to the specific loader

    Returns:
        xarray.Dataset containing the loaded data

    Raises:
        ValueError: If no suitable loader can handle the file

    Examples:
        >>> # Load various file types automatically
        >>> dataset1 = load_data_file("data.csv")  # CSV file
        >>> dataset2 = load_data_file("data.xlsx") # Excel file
        >>> dataset3 = load_data_file("data.h5")   # HDF5 file
        >>> dataset4 = load_data_file("data.txt")  # Auto-detect delimiter
    """
    path = Path(path)

    # First, try the standard load_table function
    try:
        return load_table(path, **kwargs)
    except ValueError:
        pass  # Continue to fallback options

    # Try specific loaders for common scientific formats
    extension = path.suffix.lower().lstrip(".")

    # Excel formats
    if extension in ["xlsx", "xls", "xlsm", "xlsb"]:
        return _load_excel(path, **kwargs)

    # HDF5 formats
    elif extension in ["h5", "hdf5", "hdf"]:
        return _load_hdf5(path, **kwargs)

    # Text files with unknown delimiters
    elif extension in ["txt", "dat", "asc", "prn"]:
        return _load_auto_delimited(path, **kwargs)

    # Last resort: try as delimited text
    else:
        try:
            return _load_auto_delimited(path, **kwargs)
        except Exception as exc:
            raise ValueError(f"Could not load file '{path}' with any available loader") from exc


def get_file_info(path: str | Path) -> Dict[str, Any]:
    """Get basic information about a data file without fully loading it.

    Args:
        path: Path to the data file

    Returns:
        Dictionary with file information including size, format, columns, etc.
    """
    path = Path(path)
    info = {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "extension": path.suffix.lower(),
        "exists": path.exists(),
    }

    if not path.exists():
        return info

    # Try to get column information for supported formats
    try:
        if info["extension"] in [".csv", ".tsv", ".txt"]:
            # Read just the header
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                delimiter = _auto_detect_delimiter(path)
                columns = first_line.split(delimiter)
                info["columns"] = [col.strip() for col in columns]
                info["n_columns"] = len(columns)

        elif info["extension"] in [".json"]:
            # Try to load and analyze JSON structure
            data = json.loads(path.read_text())
            if isinstance(data, list) and data and isinstance(data[0], dict):
                info["columns"] = list(data[0].keys())
                info["n_columns"] = len(data[0])
                info["n_rows"] = len(data)

    except Exception:
        # If we can't analyze, just note it
        info["analysis_error"] = "Could not analyze file structure"

    return info


def list_supported_formats() -> Dict[str, str]:
    """Return a dictionary of supported file formats and their descriptions."""
    return {
        "csv": "Comma-separated values",
        "tsv": "Tab-separated values",
        "json": "JSON table format",
        "nc/nc4/netcdf": "NetCDF format",
        "xlsx/xls": "Excel spreadsheet (requires pandas)",
        "h5/hdf5/hdf": "HDF5 format",
        "txt/dat/asc": "Text files with auto-detected delimiters",
    }


# Register additional loaders
_LOADER_MAP.update({
    "xlsx": _load_excel,
    "xls": _load_excel,
    "xlsm": _load_excel,
    "xlsb": _load_excel,
    "h5": _load_hdf5,
    "hdf5": _load_hdf5,
    "hdf": _load_hdf5,
    "txt": _load_auto_delimited,
    "dat": _load_auto_delimited,
    "asc": _load_auto_delimited,
    "prn": _load_auto_delimited,
})


__all__ = [
    "Loader",
    "get_file_info",
    "list_supported_formats",
    "load_data_file",
    "load_table",
    "register_loader",
]
