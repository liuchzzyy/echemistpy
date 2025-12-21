"""Top-level package for echemistpy.

This module wires together the public API so that the library can be consumed by
notebook users as well as more structured applications.
"""

from .core import (
    Measurement,
    MeasurementInfo,
    RawData,
    RawDataInfo,
    AnalysisResult,
    AnalysisResultInfo,
)
from .io import (
    load_data_file,
    save_measurement,
    save_results,
)
from .processing import (
    TechniqueRegistry,
    create_default_registry,
)
from .pipelines import (
    AnalysisPipeline,
)

__all__ = [
    # Core data structures
    "Measurement",
    "MeasurementInfo",
    "RawData",
    "RawDataInfo",
    "AnalysisResult",
    "AnalysisResultInfo",
    # I/O functions
    "load_data_file",
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
