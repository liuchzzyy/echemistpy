"""High-level pipeline orchestration for echemistpy.

This module provides tools for coordinating data loading, analysis, and result
aggregation across multiple measurements.
"""

from .orchestrator import AnalysisPipeline

__all__ = [
    "AnalysisPipeline",
]
