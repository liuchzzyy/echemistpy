"""Expose built-in analyzers and registries (deprecated).

This module is maintained for backward compatibility only.
New code should use echemistpy.processing.analyzers instead.

Example:
    from echemistpy.processing.analyzers import TechniqueRegistry, create_default_registry
"""

# Backward compatibility: re-export from new locations
from echemistpy.processing.analyzers import (
    CyclicVoltammetryAnalyzer,
    TechniqueRegistry,
    create_default_registry,
    TGAAnalyzer,
    XPSAnalyzer,
    XRDPowderAnalyzer,
)

__all__ = [
    "CyclicVoltammetryAnalyzer",
    "TechniqueRegistry",
    "create_default_registry",
    "TGAAnalyzer",
    "XPSAnalyzer",
    "XRDPowderAnalyzer",
]


    "TGAAnalyzer",
    "XPSAnalyzer",
    "XRDPowderAnalyzer",
    "TechniqueRegistry",
    "create_default_registry",
]
