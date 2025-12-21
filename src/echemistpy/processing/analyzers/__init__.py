"""Technique-specific analyzers module.

This module contains analyzer implementations for different measurement techniques:
- XRD: X-ray diffraction
- XPS: X-ray photoelectron spectroscopy
- TGA: Thermogravimetric analysis
- Echem: Electrochemistry (CV, etc.)

The analyzers are registered in a TechniqueRegistry for plugin-style access.
"""

from .base import TechniqueAnalyzer
from .registry import TechniqueRegistry, create_default_registry
from .echem import CyclicVoltammetryAnalyzer
from .tga import TGAAnalyzer
from .xps import XPSAnalyzer
from .xrd import XRDPowderAnalyzer

__all__ = [
    "TechniqueAnalyzer",
    "TechniqueRegistry",
    "create_default_registry",
    "CyclicVoltammetryAnalyzer",
    "TGAAnalyzer",
    "XPSAnalyzer",
    "XRDPowderAnalyzer",
]
