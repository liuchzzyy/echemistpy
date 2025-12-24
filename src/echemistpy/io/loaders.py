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
    standardize_names,
)
from echemistpy.io.structures import (
    RawData,
    RawDataInfo,
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

try:
    from echemistpy.io.plugins.xrd.MSPD import MSPDReader
except ImportError:
    MSPDReader = None  # type: ignore

if TYPE_CHECKING:
    pass


def load(
    path: str | Path,
    fmt: Optional[str] = None,
    technique: Optional[str | list[str]] = None,
    sample_name: Optional[str] = None,
    start_time: Optional[str] = None,
    instrument: Optional[str] = None,
    operator: Optional[str] = None,
    active_material_mass: Optional[str] = None,
    wave_number: Optional[str] = None,
    standardize: bool = True,
    **kwargs: Any,
) -> Tuple[RawData, RawDataInfo]:
    """Load a data file and return standardized RawData and RawDataInfo.

    Args:
        path: Path to the data file
        fmt: Optional format override (e.g., '.mpt')
        technique: Optional technique hint (string or list of strings)
        sample_name: Optional sample name override
        start_time: Optional start time override
        instrument: Optional instrument name override
        operator: Optional operator name override
        active_material_mass: Optional active material mass override
        wave_number: Optional wave number override
        standardize: Whether to automatically standardize the data (default: True)
        **kwargs: Additional arguments passed to the specific reader

    Returns:
        Tuple of (RawData, RawDataInfo)
    """
    path = Path(path) if isinstance(path, str) else path
    ext = fmt if fmt else path.suffix.lower()

    pm = get_plugin_manager()
    reader_class = pm.get_loader(ext)

    if reader_class is None:
        raise ValueError(f"No loader registered for extension: {ext}")

    # Instantiate reader and load raw data
    # Pass standard metadata to reader if provided
    standard_metadata = {
        "sample_name": sample_name,
        "start_time": start_time,
        "instrument": instrument,
        "operator": operator,
        "active_material_mass": active_material_mass,
        "wave_number": wave_number,
    }
    for k, v in standard_metadata.items():
        if v is not None:
            kwargs[k] = v

    reader = reader_class(filepath=path, **kwargs)
    if not hasattr(reader, "load"):
        raise RuntimeError(f"Reader class {reader_class.__name__} does not implement 'load' method yet.")

    raw_data, raw_info = reader.load()

    # Apply manual overrides if provided
    overrides = {
        "sample_name": sample_name,
        "start_time": start_time,
        "instrument": instrument,
        "operator": operator,
        "wave_number": wave_number,
        "active_material_mass": active_material_mass,
    }
    if technique:
        overrides["technique"] = [technique] if isinstance(technique, str) else technique

    # Filter out None values
    raw_info.update({k: v for k, v in overrides.items() if v is not None})

    if not standardize:
        return raw_data, raw_info

    # Auto-standardize
    standardized_data, standardized_info = standardize_names(raw_data, raw_info, technique_hint=technique)

    return standardized_data, standardized_info


def _register_loader(extensions: list[str], loader_class: Any) -> None:
    """Register a new loader class for specific extensions.

    Args:
        extensions: List of file extensions (e.g., ['.mpt', '.xlsx'])
        loader_class: The class to handle these files
    """
    pm = get_plugin_manager()
    pm.register_loader(extensions, loader_class)


# ============================================================================
# Utility Functions
# ============================================================================


def list_supported_formats() -> Dict[str, str]:
    """Return a dictionary of supported file formats and their descriptions."""
    pm = get_plugin_manager()
    loaders = pm.get_supported_loaders()

    formats = {}
    for ext, plugin_name in loaders.items():
        if "biologic" in plugin_name.lower():
            formats[ext] = "BioLogic EC-Lab files (.mpt)"
        elif "lanhe" in plugin_name.lower():
            formats[ext] = "LANHE battery test files (.xlsx)"
        elif "mspd" in plugin_name.lower():
            formats[ext] = "MSPD XRD files (.xye)"
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
        pm.register_loader([".mpt"], BiologicMPTReader)

    if LanheXLSXReader is not None:
        pm.register_loader([".xlsx"], LanheXLSXReader)

    if MSPDReader is not None:
        pm.register_loader([".xye"], MSPDReader)

    pm.initialized = True


# Initialize plugins on module import
_initialize_default_plugins()


__all__ = [
    "list_supported_formats",
    "load",
]
