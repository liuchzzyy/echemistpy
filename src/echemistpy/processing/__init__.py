"""Data processing module for echemistpy.

This module contains data processing, analysis, and orchestration tools:
- analyzers: Technique-specific analyzers and registry
- normalization: Data normalization and preprocessing
"""

from .analyzers import TechniqueRegistry, create_default_registry
from .analyzers import (
    TechniqueAnalyzer,
    CyclicVoltammetryAnalyzer,
    TGAAnalyzer,
    XPSAnalyzer,
    XRDPowderAnalyzer,
)

__all__ = [
    "TechniqueRegistry",
    "create_default_registry",
    "TechniqueAnalyzer",
    "CyclicVoltammetryAnalyzer",
    "TGAAnalyzer",
    "XPSAnalyzer",
    "XRDPowderAnalyzer",
]
