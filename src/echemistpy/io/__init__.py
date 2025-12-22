"""I/O module for echemistpy with plugin-based loading and saving.

This module provides a unified interface for loading and saving scientific
measurement data. It uses a plugin system to support multiple file formats
and can be easily extended with custom readers and writers.

Main Functions:
    - load(): Load data files (RawData + RawDataInfo)
    - save(): Save Measurement data
    - standardize_measurement(): Convert RawData to Measurement
    - register_loader(): Add custom file format support

Data Structures:
    - RawData, RawDataInfo: Unprocessed data from files
    - Measurement, MeasurementInfo: Standardized measurement data
    - AnalysisResult, AnalysisResultInfo: Processed analysis results
"""

from .loaders import (
    DataStandardizer,
    detect_technique,
    get_file_info,
    list_supported_formats,
    load,
    load_data_file,  # Backward compatibility
    load_table,  # Backward compatibility
    register_loader,
    standardize_measurement,
)
from .plugin_manager import get_plugin_manager
from .saver import save, save_measurement, save_results

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
    # Data structures
    "AnalysisResult",
    "AnalysisResultInfo",
    "Measurement",
    "MeasurementInfo",
    "RawData",
    "RawDataInfo",
    # Standardization utilities
    "DataStandardizer",
    "detect_technique",
    "standardize_measurement",
    # File utilities
    "get_file_info",
    "list_supported_formats",
    # Plugin management
    "get_plugin_manager",
    "register_loader",
    # Loading functions
    "load",
    "load_data_file",  # Backward compatibility
    "load_table",  # Backward compatibility
    # Saving functions
    "save",
    "save_measurement",
    "save_results",
]

