"""
Utilities module providing common helper functions.

This module contains utility functions for data processing, validation,
and transformation used across the package.
"""

from echemistpy.utils.data_processing import baseline_correction, normalize, smooth
from echemistpy.utils.validation import check_dimensions, validate_data

__all__ = [
    "baseline_correction",
    "check_dimensions",
    "normalize",
    "smooth",
    "validate_data",
]
