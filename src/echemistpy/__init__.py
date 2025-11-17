"""Top-level package for echemistpy.

This module wires together the public API so that the library can be consumed by
notebook users as well as more structured applications.  The key primitives are

* :class:`Measurement` - tabular data with metadata and axis definitions.
* :class:`TechniqueAnalyzer` - pluggable strategy object for domain specific
  analytics (XRD, XPS, TGA, ...).
* :class:`AnalysisPipeline` - orchestrates I/O, metadata management and analysis
  execution.

The package ships with a default registry that already knows about several
standard techniques.  Custom analyzers can be added by registering them with the
same registry instance.
"""

from .analysis import TechniqueRegistry, create_default_registry
from .io import AnalysisResult, Axis, Measurement, MeasurementMetadata
from .pipelines.manager import AnalysisPipeline

__all__ = [
    "AnalysisPipeline",
    "AnalysisResult",
    "Axis",
    "Measurement",
    "MeasurementMetadata",
    "TechniqueRegistry",
    "create_default_registry",
]

# Initialize a default registry with the analyzers that ship with the package so
# users can simply import ``echemistpy`` and immediately analyze data.
default_registry = create_default_registry()
__all__.append("default_registry")
