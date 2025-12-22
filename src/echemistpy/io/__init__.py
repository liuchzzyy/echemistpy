"""I/O helpers with pluggy-based plugin system."""

from .loaders import (
    DataStandardizer,
    detect_technique,
    get_file_info,
    list_supported_formats,
    load_data_file,
    load_table,
    register_loader,
    standardize_measurement,
)
from .plugin_manager import get_plugin_manager
from .saver import save_measurement, save_results

# Import core structures
from .structures import (
    AnalysisResult,
    AnalysisResultInfo,
    Measurement,
    MeasurementInfo,
    RawData,
    RawDataInfo,
)

__all__ = [
    "AnalysisResult",
    "AnalysisResultInfo",
    "DataStandardizer",
    # Data structures
    "Measurement",
    "MeasurementInfo",
    "RawData",
    "RawDataInfo",
    "detect_technique",
    "get_file_info",
    # Plugin management
    "get_plugin_manager",
    "list_supported_formats",
    # Loading functions
    "load_data_file",
    "load_table",
    "register_loader",
    # Saving functions
    "save_measurement",
    "save_results",
    # Standardization
    "standardize_measurement",
]
