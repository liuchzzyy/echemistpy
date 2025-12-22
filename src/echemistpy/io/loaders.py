"""Unified file loading interface for scientific measurements.

This module provides a simplified interface for loading data files using
the plugin system. It automatically detects file formats and delegates
loading to the appropriate plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from echemistpy.io.plugin_manager import get_plugin_manager
from echemistpy.io.standardizer import (
    DataStandardizer,
    detect_technique,
    standardize_measurement,
)
from echemistpy.io.structures import (
    RawData,
    RawDataInfo,
)

# Type alias for backward compatibility
Loader = Callable[[Path, Any], Tuple[RawData, RawDataInfo]]


# ============================================================================
# Plugin System Setup
# ============================================================================


def _initialize_default_plugins() -> None:
    """Initialize and register default loader and saver plugins."""
    pm = get_plugin_manager()

    # Register echem-specific plugins
    from echemistpy.io.plugings.echem import BiologicLoaderPlugin, LanheLoaderPlugin
    pm.register_plugin(BiologicLoaderPlugin(), name="biologic")
    pm.register_plugin(LanheLoaderPlugin(), name="lanhe")

    # Register generic format plugins
    from echemistpy.io.plugings.generic_loaders import (
        CSVLoaderPlugin,
        ExcelLoaderPlugin,
        HDF5LoaderPlugin,
    )
    pm.register_plugin(CSVLoaderPlugin(), name="csv")
    pm.register_plugin(ExcelLoaderPlugin(), name="excel")
    pm.register_plugin(HDF5LoaderPlugin(), name="hdf5")

    # Register saver plugins
    from echemistpy.io.plugings.generic_savers import (
        CSVSaverPlugin,
        HDF5SaverPlugin,
        JSONSaverPlugin,
    )
    pm.register_plugin(CSVSaverPlugin(), name="csv_saver")
    pm.register_plugin(HDF5SaverPlugin(), name="hdf5_saver")
    pm.register_plugin(JSONSaverPlugin(), name="json_saver")


# Initialize plugins on module import
_initialize_default_plugins()


# ============================================================================
# Unified Loader Interface
# ============================================================================


def load(path: str | Path, fmt: Optional[str] = None, **kwargs: Any) -> Tuple[RawData, RawDataInfo]:
    """Load a data file using the plugin system.

    This is the main entry point for loading data files. It automatically
    detects the file format and uses the appropriate plugin to load the data.

    Args:
        path: Path to the data file
        fmt: Optional format override (file extension without dot)
        **kwargs: Additional arguments passed to the specific loader plugin

    Returns:
        Tuple of (RawData, RawDataInfo) containing loaded data and metadata

    Raises:
        ValueError: If file format is not supported

    Example:
        >>> raw_data, raw_info = load("data.mpt")
        >>> raw_data, raw_info = load("data.xlsx", fmt="xlsx")
    """
    path = Path(path)
    pm = get_plugin_manager()
    return pm.load_file(path, fmt=fmt, **kwargs)


def register_loader(plugin: Any, name: Optional[str] = None) -> None:
    """Register a custom loader plugin.

    Args:
        plugin: Plugin instance that implements LoaderSpec interface
        name: Optional plugin name

    Example:
        >>> from echemistpy.io.plugin_specs import hookimpl
        >>> from traitlets import HasTraits
        >>>
        >>> class MyLoaderPlugin(HasTraits):
        ...     @hookimpl
        ...     def get_supported_extensions(self):
        ...         return ["xyz"]
        ...
        ...     @hookimpl
        ...     def load_file(self, filepath, **kwargs):
        ...         # Custom loading logic
        ...         dataset = ...  # Create xarray Dataset
        ...         raw_data = RawData(data=dataset)
        ...         raw_data_info = RawDataInfo(meta={"technique": "Custom"})
        ...         return raw_data, raw_data_info
        >>>
        >>> register_loader(MyLoaderPlugin(), name="my_loader")
    """
    pm = get_plugin_manager()
    pm.register_plugin(plugin, name=name)


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
    supported_extensions = list(pm.get_supported_loaders().keys())

    info = {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "extension": path.suffix.lower(),
        "exists": path.exists(),
        "supported": path.suffix.lower().lstrip(".") in supported_extensions,
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

_plugins_initialized = False


def _initialize_default_plugins() -> None:
    """Initialize and register default loader and saver plugins."""
    global _plugins_initialized
    
    if _plugins_initialized:
        return
    
    pm = get_plugin_manager()

    # Register echem-specific plugins
    from echemistpy.io.plugings.echem import BiologicLoaderPlugin, LanheLoaderPlugin
    pm.register_plugin(BiologicLoaderPlugin(), name="biologic")
    pm.register_plugin(LanheLoaderPlugin(), name="lanhe")

    # Register generic format plugins
    from echemistpy.io.plugings.generic_loaders import (
        CSVLoaderPlugin,
        ExcelLoaderPlugin,
        HDF5LoaderPlugin,
    )
    pm.register_plugin(CSVLoaderPlugin(), name="csv")
    pm.register_plugin(ExcelLoaderPlugin(), name="excel")
    pm.register_plugin(HDF5LoaderPlugin(), name="hdf5")

    # Register saver plugins
    from echemistpy.io.plugings.generic_savers import (
        CSVSaverPlugin,
        HDF5SaverPlugin,
        JSONSaverPlugin,
    )
    pm.register_plugin(CSVSaverPlugin(), name="csv_saver")
    pm.register_plugin(HDF5SaverPlugin(), name="hdf5_saver")
    pm.register_plugin(JSONSaverPlugin(), name="json_saver")
    
    _plugins_initialized = True


# Initialize plugins on module import
_initialize_default_plugins()


__all__ = [
    "DataStandardizer",
    "Loader",
    "detect_technique",
    "get_file_info",
    "get_plugin_manager",
    "list_supported_formats",
    "load",
    "register_loader",
    "standardize_measurement",
]

# Backward compatibility aliases
load_data_file = load
load_table = load

