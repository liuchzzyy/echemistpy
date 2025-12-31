"""Registries that keep track of available analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import xarray as xr
from traitlets import HasTraits, Instance, List as TList, Unicode

from echemistpy.io.structures import RawData, RawDataInfo, ResultsData, ResultsDataInfo

if TYPE_CHECKING:
    pass


class TechniqueAnalyzer(HasTraits, ABC):
    """Template used by all built-in analyzers."""

    technique = Unicode(help="Technique identifier")
    instrument = Unicode(None, allow_none=True, help="Instrument identifier")
    name = Unicode(help="Analyzer name")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.name:
            self.name = self.__class__.__name__

    def analyze(
        self,
        raw_data: RawData,
        raw_info: Optional[RawDataInfo] = None,
        **kwargs: Any,
    ) -> Tuple[ResultsData, ResultsDataInfo]:
        """Perform the full analysis workflow.

        This includes validation, preprocessing, computation, and packaging.
        Metadata from raw_info is carried over to the results.

        Args:
            raw_data: Standardized raw data container
            raw_info: Optional metadata from the raw data
            **kwargs: Additional parameters to store in ResultsDataInfo

        Returns:
            Tuple of (ResultsData, ResultsDataInfo)
        """
        # 1. Validate
        self.validate(raw_data)

        # 2. Preprocess (on a copy to avoid side effects)
        cleaned = self.preprocess(raw_data.copy())

        # 3. Compute
        summary, tables = self.compute(cleaned)

        # 4. Package results
        # Start with metadata from raw_info if provided
        info_dict = raw_info.to_dict() if raw_info else {}

        # Update with analyzer-specific info
        info_dict.update({
            "technique": [self.technique],
            "parameters": summary,
        })
        # Add any extra kwargs
        info_dict.update(kwargs)

        info = ResultsDataInfo(**info_dict)

        if not tables:
            data = xr.Dataset()
        elif len(tables) == 1:
            data = next(iter(tables.values()))
        else:
            # Create a DataTree if multiple tables are returned
            data = xr.DataTree.from_dict(tables)

        return ResultsData(data=data), info

    @property
    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Columns that must be present in the data."""

    def validate(self, raw_data: RawData) -> None:
        """Check if raw_data contains all required columns.

        Args:
            raw_data: RawData instance to validate

        Raises:
            ValueError: If any required columns are missing
        """
        # For now, we check the root dataset variables and coordinates
        available = set(raw_data.variables) | set(raw_data.coords)
        missing = [col for col in self.required_columns if col not in available]
        if missing:
            raise ValueError(f"Analyzer '{self.name}' requires columns {self.required_columns}, but {missing} are missing from the data.")

    def preprocess(self, raw_data: RawData) -> RawData:
        """Optional preprocessing step (e.g., filtering, normalization)."""
        return raw_data

    @abstractmethod
    def compute(self, raw_data: RawData) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        """Perform the main calculation and return summary + tables.

        Returns:
            A tuple containing:
            - summary: Dictionary of scalar results/parameters
            - tables: Dictionary mapping names to xarray.Dataset objects
        """


class TechniqueRegistry(HasTraits):
    """Map technique and instrument identifiers to analyzer instances."""

    _analyzers = TList(Instance(TechniqueAnalyzer), help="Internal list of registered analyzers")

    def register(self, analyzer: TechniqueAnalyzer) -> None:
        """Register an analyzer instance.

        Args:
            analyzer: TechniqueAnalyzer instance
        """
        if analyzer not in self._analyzers:
            self._analyzers.append(analyzer)

    def unregister(self, analyzer: TechniqueAnalyzer) -> None:
        """Unregister an analyzer instance.

        Args:
            analyzer: TechniqueAnalyzer instance
        """
        if analyzer in self._analyzers:
            self._analyzers.remove(analyzer)

    def get_analyzer(self, technique: str, instrument: Optional[str] = None) -> TechniqueAnalyzer:
        """Get analyzer for a technique and optionally an instrument.

        Args:
            technique: Technique identifier (case-insensitive)
            instrument: Optional instrument identifier (case-insensitive)

        Returns:
            TechniqueAnalyzer instance

        Raises:
            KeyError: If no matching analyzer is found
        """
        tech_lower = technique.lower()
        inst_lower = instrument.lower() if instrument else None

        # 1. Try specific instrument match
        if inst_lower:
            for a in self._analyzers:
                if a.technique.lower() == tech_lower and a.instrument and a.instrument.lower() == inst_lower:
                    return a

        # 2. Try generic technique match (no instrument specified in analyzer)
        for a in self._analyzers:
            if a.technique.lower() == tech_lower and not a.instrument:
                return a

        # 3. Fallback to first technique match
        for a in self._analyzers:
            if a.technique.lower() == tech_lower:
                return a

        raise KeyError(f"No analyzer registered for technique '{technique}'" + (f" and instrument '{instrument}'" if instrument else ""))

    def available(self) -> List[str]:
        """Get list of registered techniques.

        Returns:
            List of available technique identifiers
        """
        return sorted({a.technique for a in self._analyzers})

    def __contains__(self, technique: str) -> bool:
        """Check if technique is registered."""
        return any(a.technique.lower() == technique.lower() for a in self._analyzers)

    def __len__(self) -> int:
        """Get number of registered analyzers."""
        return len(self._analyzers)


def create_default_registry() -> TechniqueRegistry:
    """Return a registry populated with the built-in analyzers.

    Returns:
        TechniqueRegistry with standard analyzers
    """
    from .echem import CyclicVoltammetryAnalyzer

    registry = TechniqueRegistry()
    registry.register(CyclicVoltammetryAnalyzer())
    return registry
