"""Base classes shared across technique-specific analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import xarray as xr

from echemistpy.io.structures import AnalysisResult, SummaryData


class TechniqueAnalyzer(ABC):
    """Template used by all built-in analyzers."""

    technique: str

    def __init__(self, *, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def analyze(self, summary_data: SummaryData) -> AnalysisResult:
        cleaned = self.preprocess(summary_data.copy())
        summary, tables = self.compute(cleaned)
        return AnalysisResult(
            data=xr.Dataset(),  # Initialize with empty dataset
        )

    @property
    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Columns that must be present in the summary data."""

    def preprocess(self, summary_data: SummaryData) -> SummaryData:
        return summary_data

    @abstractmethod
    def compute(self, summary_data: SummaryData) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        """Perform the main calculation and return summary + tables."""
