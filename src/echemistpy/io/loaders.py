"""Unified file loading interface for scientific measurements.

This module provides a simplified interface for loading data files.
It automatically detects file formats and delegates loading to the 
appropriate reader in the plugins directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from echemistpy.io.plugin_manager import get_plugin_manager
from echemistpy.io.standardizer import (
    detect_technique,
    standardize_measurement,
)
from echemistpy.io.structures import (
    Measurement,
    RawData,
    RawDataInfo,
)


def _initialize_loaders() -> None:
    """Register all available loaders to the plugin manager."""
    pm = get_plugin_manager()

    # 1. BioLogic MPT Loader
    try:
        from echemistpy.io.plugins.echem.BiologicMPTReader import BiologicDataReader
        pm.register_loader([".mpt"], BiologicDataReader)
    except ImportError:
        pass

    # 2. LANHE XLSX Loader
    try:
        from echemistpy.io.plugins.echem.LanheXLSXReader import XLSXDataReader
        pm.register_loader([".xlsx"], XLSXDataReader)
    except ImportError:
        pass


# Initialize on module import
_initialize_loaders()


def load_data_file(
    path: str | Path, 
    fmt: Optional[str] = None, 
    **kwargs: Any
) -> Tuple[RawData, RawDataInfo]:
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

    # 统一调用逻辑：
    # 假设所有 Reader 都有一个 load() 静态方法或类方法
    # 或者我们需要在这里根据不同的类做一些适配
    
    # 针对目前的 BiologicDataReader 和 XLSXDataReader 的适配逻辑
    if hasattr(reader_class, "load"):
        # 现有的类通常返回 (metadata, data_dict)
        metadata, data_dict = reader_class.load(path, **kwargs)
        
        # 将其包装为 RawData (xarray) 和 RawDataInfo
        import xarray as xr
        ds = xr.Dataset({k: (("row",), v) for k, v in data_dict.items()})
        
        raw_data = RawData(data=ds)
        raw_data_info = RawDataInfo(meta=metadata)
        return raw_data, raw_data_info
    
    raise RuntimeError(f"Reader class {reader_class.__name__} does not implement 'load' method")


def load_measurement(
    path: str | Path, 
    technique: Optional[str] = None,
    **kwargs: Any
) -> Measurement:
    """Load a file and return a standardized Measurement object.

    This is the high-level function that combines loading and standardization.
    """
    raw_data, raw_info = load_data_file(path, **kwargs)
    
    if technique is None:
        technique = detect_technique(raw_data, raw_info)
        
    return standardize_measurement(raw_data, raw_info, technique=technique)


__all__ = [
    "load_data_file",
    "load_measurement",
]

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

