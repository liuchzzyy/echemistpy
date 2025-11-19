"""Top-level package for echemistpy.

This module wires together the public API so that the library can be consumed by
notebook users as well as more structured applications.
"""

# from .analysis import TechniqueRegistry, create_default_registry  # Module missing
from .io import (
    Measurement, 
    MeasurementInfo, 
    Results, 
    ResultsInfo, 
    load_data_file, 
    save_measurement, 
    save_results
)
# from .pipelines.manager import AnalysisPipeline

__all__ = [
    # "AnalysisPipeline",
    "Measurement",
    "MeasurementInfo",
    "Results",
    "ResultsInfo",
    "load_data_file",
    "save_measurement",
    "save_results",
]

# Initialize a default registry with the analyzers that ship with the package so
# users can simply import ``echemistpy`` and immediately analyze data.
# default_registry = create_default_registry()
# __all__.append("default_registry")
