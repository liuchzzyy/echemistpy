"""Top-level package for echemistpy.

This module wires together the public API so that the library can be consumed by
notebook users as well as more structured applications.
"""

from .io import (
    AnalysisResult,
    AnalysisResultInfo,
    Measurement,
    MeasurementInfo,
    RawData,
    RawDataInfo,
    load,
    load_data_file,  # Backward compatibility
    save,
    save_measurement,
    save_results,
)
from .pipelines import AnalysisPipeline
from .processing import TechniqueRegistry, create_default_registry

__all__ = [
    # Core data structures
    "Measurement",
    "MeasurementInfo",
    "RawData",
    "RawDataInfo",
    "AnalysisResult",
    "AnalysisResultInfo",
    # I/O functions
    "load",
    "load_data_file",  # Backward compatibility
    "save",
    "save_measurement",
    "save_results",
    # Processing
    "TechniqueRegistry",
    "create_default_registry",
    # Pipelines
    "AnalysisPipeline",
]

# Initialize a default registry with the analyzers that ship with the package
default_registry = create_default_registry()
__all__.append("default_registry")

