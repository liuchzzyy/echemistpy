"""I/O helpers with pluggy-based plugin system."""

from .loaders import (
    load_data_file,
    load_table,
    register_loader,
    get_file_info,
    list_supported_formats,
    standardize_measurement,
    detect_technique,
    DataStandardizer,
)
from .saver import save_measurement, save_results
from .plugin_manager import get_plugin_manager

# Import core structures
from .structures import (
    Measurement,
    MeasurementInfo,
    RawData,
    RawDataInfo,
    AnalysisResult,
    AnalysisResultInfo,
)

__all__ = [
    # Data structures
    "Measurement",
    "MeasurementInfo",
    "RawData",
    "RawDataInfo",
    "AnalysisResult",
    "AnalysisResultInfo",
    # Loading functions
    "load_data_file",
    "load_table",
    "register_loader",
    "get_file_info",
    "list_supported_formats",
    # Standardization
    "standardize_measurement",
    "detect_technique",
    "DataStandardizer",
    # Saving functions
    "save_measurement",
    "save_results",
    # Plugin management
    "get_plugin_manager",
]
