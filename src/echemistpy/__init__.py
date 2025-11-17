"""Top-level package for the eChemistPy toolkit."""

from . import analysis, external, io, math, ploting
from .io.reorganization import DataObject, MeasurementMetadata, MeasurementRecord
from .io import initialize_analysis_plugins, load_builtin_readers

load_builtin_readers()
initialize_analysis_plugins()

__all__ = [
    "DataObject",
    "MeasurementMetadata",
    "MeasurementRecord",
    "analysis",
    "external",
    "io",
    "math",
    "ploting",
]
