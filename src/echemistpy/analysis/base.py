"""Base classes shared across technique-specific analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import xarray as xr

from echemistpy.io.structures import AnalysisResult, Measurement


class TechniqueAnalyzer(ABC):
    """Template used by all built-in analyzers."""

    technique: str

    def __init__(self, *, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def analyze(self, measurement: Measurement) -> AnalysisResult:
        measurement.require_columns(self.required_columns)
        cleaned = self.preprocess(measurement.copy())
        summary, tables = self.compute(cleaned)
        return AnalysisResult(
            technique=self.technique,
            sample_name=measurement.metadata.sample_name,
            summary=summary,
            tables=tables,
        )

    @property
    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Columns that must be present in the measurement data."""

    def preprocess(self, measurement: Measurement) -> Measurement:
        return measurement

    @abstractmethod
    def compute(self, measurement: Measurement) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        """Perform the main calculation and return summary + tables."""
