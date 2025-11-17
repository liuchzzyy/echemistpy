"""
echemistpy - A Python package for electrochemistry characterization analysis.

This package provides tools and utilities for analyzing various electrochemical
and materials characterization techniques including:
- Electrochemistry (Echem)
- Electrochemical Quartz Crystal Microbalance (EQCM)
- X-ray Diffraction (XRD)
- X-ray Photoelectron Spectroscopy (XPS)
- X-ray Absorption Spectroscopy (XAS)
- Transmission Electron Microscopy (TEM)
- Scanning Electron Microscopy (SEM)
- Scanning Transmission X-ray Microscopy (STXM)
- Thermogravimetric Analysis (TGA)
- Inductively Coupled Plasma Optical Emission Spectrometry (ICP-OES)
"""

__version__ = "0.1.0"
__author__ = "Cheng Liu, PhD"

# Import main submodules for convenience
from echemistpy import core, io, techniques, utils, visualization

__all__ = [
    "__author__",
    "__version__",
    "core",
    "io",
    "techniques",
    "utils",
    "visualization",
]
