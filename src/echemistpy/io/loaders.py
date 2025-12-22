"""Unified file loading interface for scientific data.

This module provides a simplified interface for loading data files.
It automatically detects file formats and delegates loading to the
appropriate reader in the plugins directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from echemistpy.io.plugin_manager import get_plugin_manager
from echemistpy.io.standardizer import (
    detect_technique,
    standardize_rawdata,
)
from echemistpy.io.structures import (
    RawData,
    RawDataInfo,
    SummaryData,
)

# Import default plugins
try:
    from echemistpy.io.plugins.echem.BiologicMPTReader import BiologicMPTReader
except ImportError:
    BiologicMPTReader = None  # type: ignore

try:
    from echemistpy.io.plugins.echem.LanheXLSXReader import LanheXLSXReader
except ImportError:
    LanheXLSXReader = None  # type: ignore

if TYPE_CHECKING:
    pass


def load(path: str | Path, fmt: Optional[str] = None, **kwargs: Any) -> Tuple[RawData, RawDataInfo]:
    """Load a raw data file and return RawData and RawDataInfo.

    Args:
        path: Path to the data file
        fmt: Optional format override (e.g., '.mpt')
        **kwargs: Additional arguments passed to the reader

    Returns:
        Tuple of (RawData, RawDataInfo)
    """
    path = Path(path)
    ext = fmt if fmt else path.suffix.lower()

    pm = get_plugin_manager()
    reader_class = pm.get_loader(ext)

    if reader_class is None:
        raise ValueError(f"No loader registered for extension: {ext}")

    # Instantiate reader and load data
    reader = reader_class(filepath=path, **kwargs)
    if hasattr(reader, "load"):
        return reader.load()

    raise RuntimeError(f"Reader class {reader_class.__name__} does not implement 'load' method")


def load_summary(path: str | Path, technique: Optional[str] = None, **kwargs: Any) -> Tuple[SummaryData, Any]:
    """Load a file and return a standardized SummaryData object.

    This is the high-level function that combines loading and standardization.
    """
    raw_data, raw_info = load(path, **kwargs)

    if technique is None:
        technique = detect_technique(raw_data.data)

    return standardize_rawdata(raw_data, raw_info, technique_hint=technique)


def register_loader(extensions: list[str], loader_class: Any) -> None:
    """Register a new loader class for specific extensions.

    Args:
        extensions: List of file extensions (e.g., ['.mpt', '.mpr'])
        loader_class: The class to handle these files
    """
    pm = get_plugin_manager()
    pm.register_loader(extensions, loader_class)


# ============================================================================
# Utility Functions
# ============================================================================


def get_file_info(path: str | Path) -> Dict[str, Any]:
    """Get basic information about a data file without fully loading it.

    Args:
        path: Path to the data file

    Returns:
        Dictionary with file information including size, format, etc.
    """
    path = Path(path)
    pm = get_plugin_manager()
    supported_extensions = pm.list_supported_extensions()

    info = {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "extension": path.suffix.lower(),
        "exists": path.exists(),
        "supported": path.suffix.lower() in supported_extensions,
    }

    if not path.exists():
        return info

    # Try to get column information for supported formats
    try:
        if info["extension"] in {".csv", ".tsv"}:
            # Read just the header
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                delimiter = "," if info["extension"] == ".csv" else "\t"
                columns = first_line.split(delimiter)
                info["columns"] = [col.strip() for col in columns]
                info["n_columns"] = len(columns)

    except Exception:
        # If we can't analyze, just note it
        info["analysis_error"] = "Could not analyze file structure"

    return info


def list_supported_formats() -> Dict[str, str]:
    """Return a dictionary of supported file formats and their descriptions."""
    pm = get_plugin_manager()
    loaders = pm.get_supported_loaders()

    # Group by plugin type
    formats = {}
    for ext, plugin_name in loaders.items():
        if "biologic" in plugin_name.lower():
            formats[ext] = "BioLogic EC-Lab files"
        elif "lanhe" in plugin_name.lower():
            formats[ext] = "LANHE battery test files"
        elif "csv" in plugin_name.lower():
            formats[ext] = "Comma/Tab-separated values"
        elif "excel" in plugin_name.lower():
            formats[ext] = "Excel spreadsheet"
        elif "hdf5" in plugin_name.lower():
            formats[ext] = "HDF5/NetCDF format"
        else:
            formats[ext] = f"Loaded by {plugin_name}"

    return formats


# ============================================================================
# Plugin System Initialization
# ============================================================================


def _initialize_default_plugins() -> None:
    """Initialize and register default loader and saver plugins."""
    pm = get_plugin_manager()
    if pm.initialized:
        return

    # Register echem-specific plugins
    if BiologicMPTReader is not None:
        pm.register_loader([".mpt", ".mpr"], BiologicMPTReader)

    if LanheXLSXReader is not None:
        pm.register_loader([".xlsx", ".ccs"], LanheXLSXReader)

    pm.initialized = True


# Initialize plugins on module import
_initialize_default_plugins()


__all__ = [
    "detect_technique",
    "get_file_info",
    "get_plugin_manager",
    "list_supported_formats",
    "load",
    "load_summary",
    "register_loader",
    "standardize_rawdata",
]
