"""
Techniques module containing implementations for various characterization methods.

This module provides specific implementations for each characterization technique,
including:
- Electrochemistry (echem)
- X-ray techniques (xrd, xps, xas)
- Microscopy techniques (tem, sem, stxm)
- Thermal analysis (tga)
- Chemical analysis (icp_oes, eqcm)
"""

from echemistpy.techniques.echem import Electrochemistry
from echemistpy.techniques.eqcm import EQCM
from echemistpy.techniques.icp_oes import ICPOES
from echemistpy.techniques.sem import SEM
from echemistpy.techniques.stxm import STXM
from echemistpy.techniques.tem import TEM
from echemistpy.techniques.tga import TGA
from echemistpy.techniques.xas import XAS
from echemistpy.techniques.xps import XPS
from echemistpy.techniques.xrd import XRD

__all__ = [
    "EQCM",
    "ICPOES",
    "SEM",
    "STXM",
    "TEM",
    "TGA",
    "XAS",
    "XPS",
    "XRD",
    "Electrochemistry",
]
