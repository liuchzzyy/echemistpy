"""Base classes shared across technique-specific analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import xarray as xr
from traitlets import HasTraits, Unicode

from echemistpy.io.structures import RawData, ResultsData


class TechniqueAnalyzer(HasTraits, ABC):
    """Template used by all built-in analyzers."""

    technique = Unicode(help="Technique identifier")
    name = Unicode(help="Analyzer name")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.name:
            self.name = self.__class__.__name__

    def analyze(self, raw_data: RawData) -> ResultsData:
        cleaned = self.preprocess(raw_data.copy())
        summary, tables = self.compute(cleaned)
        return ResultsData(
            data=xr.Dataset(),  # Initialize with empty dataset
        )

    @property
    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Columns that must be present in the data."""

    def preprocess(self, raw_data: RawData) -> RawData:
        return raw_data

    @abstractmethod
    def compute(self, raw_data: RawData) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        """Perform the main calculation and return summary + tables."""
