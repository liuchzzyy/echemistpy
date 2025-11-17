"""
Core module providing base classes and fundamental functionality.

This module contains abstract base classes and core functionality that all
characterization techniques inherit from.
"""

from echemistpy.core.base import BaseCharacterization, BaseData
from echemistpy.core.exceptions import AnalysisError, DataLoadError, EchemistpyError

__all__ = [
    "AnalysisError",
    "BaseCharacterization",
    "BaseData",
    "DataLoadError",
    "EchemistpyError",
]
