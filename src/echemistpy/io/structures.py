"""Shared data structures based on :mod:`xarray`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

import xarray as xr


@dataclass(slots=True)
class Axis:
    """Describe a measurement axis and its sampling."""

    name: str
    unit: Optional[str] = None
    values: Optional[Sequence[float]] = None


@dataclass(slots=True)
class MeasurementMetadata:
    """Holds descriptive metadata for a measurement."""

    technique: str
    sample_name: str
    instrument: Optional[str] = None
    operator: Optional[str] = None
    extras: MutableMapping[str, Any] = field(default_factory=dict)

    def copy(self) -> "MeasurementMetadata":
        return MeasurementMetadata(
            technique=self.technique,
            sample_name=self.sample_name,
            instrument=self.instrument,
            operator=self.operator,
            extras=dict(self.extras),
        )


@dataclass(slots=True)
class Measurement:
    """Container representing the raw data from an experiment."""

    data: xr.Dataset
    metadata: MeasurementMetadata
    axes: List[Axis] = field(default_factory=list)

    def copy(self) -> "Measurement":
        return Measurement(
            data=self.data.copy(deep=True),
            metadata=self.metadata.copy(),
            axes=[Axis(axis.name, axis.unit, axis.values) for axis in self.axes],
        )

    def require_variables(self, variables: Iterable[str]) -> None:
        missing = [name for name in variables if name not in self.data.variables]
        if missing:
            raise ValueError(
                "Measurement is missing required variables: " + ", ".join(missing)
            )

    # Backwards compatible alias
    def require_columns(self, columns: Iterable[str]) -> None:  # pragma: no cover - thin wrapper
        self.require_variables(columns)


@dataclass(slots=True)
class AnalysisResult:
    """A light-weight container for processed data."""

    technique: str
    sample_name: str
    summary: Dict[str, Any]
    tables: Dict[str, xr.Dataset] = field(default_factory=dict)
    figures: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "AnalysisResult") -> "AnalysisResult":
        if self.sample_name != other.sample_name:
            raise ValueError("Cannot merge results from different samples.")
        merged_summary = {**self.summary, **other.summary}
        merged_tables = {**self.tables, **other.tables}
        merged_figures = {**self.figures, **other.figures}
        return AnalysisResult(
            technique=self.technique,
            sample_name=self.sample_name,
            summary=merged_summary,
            tables=merged_tables,
            figures=merged_figures,
        )
