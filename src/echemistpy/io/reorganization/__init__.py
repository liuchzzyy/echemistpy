"""Metadata-aware data objects, cleaning helpers, and built-in readers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import xarray as xr

from ..reading.base import BaseFileReader

_SAMPLE_DIM = "index"


@dataclass(slots=True)
class MeasurementMetadata:
    """Basic information that describes a characterization result."""

    sample_id: str
    technique: str
    instrument: str | None = None
    operator: str | None = None
    run_date: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """Return a concise description suitable for logging."""

        instrument = self.instrument or "unspecified instrument"
        operator = self.operator or "unknown operator"
        date = self.run_date.isoformat() if self.run_date else "undated"
        return (
            f"Sample {self.sample_id} measured via {self.technique} on {instrument} "
            f"({operator}, {date})"
        )


@dataclass(slots=True)
class MeasurementRecord:
    """Pair array-backed data with metadata and optional annotations."""

    metadata: MeasurementMetadata
    data: xr.Dataset
    annotations: Sequence[str] = field(default_factory=tuple)

    def iter_annotations(self) -> Iterable[str]:
        """Yield annotation strings, skipping empty entries."""

        return (note for note in self.annotations if note)

    def summary(self) -> dict[str, Any]:
        """Provide a dictionary summary useful for dashboards or logs."""

        stats: dict[str, Any] = {
            "sample_id": self.metadata.sample_id,
            "technique": self.metadata.technique,
            "row_count": int(self.data.sizes.get(_SAMPLE_DIM, 0)),
            "column_count": len(self.data.data_vars),
        }
        numeric_vars = _numeric_variables(self.data)
        if numeric_vars:
            means: dict[str, float] = {}
            stds: dict[str, float] = {}
            for name in numeric_vars:
                values = self.data[name].values.astype(float)
                if values.size == 0:
                    continue
                means[name] = float(np.mean(values))
                stds[name] = float(np.std(values, ddof=0))
            if means:
                stats["mean"] = means
            if stds:
                stats["std"] = stds
        return stats

    def to_data_object(self) -> "DataObject":
        """Convert the measurement record into a :class:`DataObject`."""

        return DataObject(metadata=self.metadata, dataset=self.data.copy(deep=True), annotations=self.annotations)


@dataclass(slots=True)
class DataObject:
    """Bundle structured data with metadata for persistent storage."""

    metadata: MeasurementMetadata
    dataset: xr.Dataset
    annotations: Sequence[str] = field(default_factory=tuple)

    @staticmethod
    def _metadata_to_dict(metadata: MeasurementMetadata) -> dict[str, Any]:
        return {
            "sample_id": metadata.sample_id,
            "technique": metadata.technique,
            "instrument": metadata.instrument,
            "operator": metadata.operator,
            "run_date": metadata.run_date.isoformat() if metadata.run_date else None,
            "extra": dict(metadata.extra),
        }

    def metadata_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the metadata."""

        return self._metadata_to_dict(self.metadata)

    def to_dataset(self, *, include_metadata: bool = True) -> xr.Dataset:
        """Return an :class:`xarray.Dataset` representation of the object."""

        dataset = self.dataset.copy(deep=True)
        if include_metadata:
            dataset.attrs["metadata"] = self.metadata_dict()
            dataset.attrs["annotations"] = list(self.annotations)
        return dataset

    def to_measurement_record(self) -> MeasurementRecord:
        """Convert the data object back into a :class:`MeasurementRecord`."""

        return MeasurementRecord(metadata=self.metadata, data=self.dataset.copy(deep=True), annotations=self.annotations)

    @classmethod
    def from_measurement_record(cls, record: MeasurementRecord) -> "DataObject":
        """Create a :class:`DataObject` from an existing record."""

        return cls(metadata=record.metadata, dataset=record.data.copy(deep=True), annotations=record.annotations)


def _numeric_variables(dataset: xr.Dataset) -> list[str]:
    numeric = []
    for name, data in dataset.data_vars.items():
        if np.issubdtype(data.dtype, np.number):
            numeric.append(name)
    return numeric


def _sample_dim(dataset: xr.Dataset) -> str:
    if _SAMPLE_DIM in dataset.dims:
        return _SAMPLE_DIM
    if dataset.dims:
        return next(iter(dataset.dims))
    return _SAMPLE_DIM


def drop_outliers(record: MeasurementRecord, zscore_threshold: float = 3.0) -> MeasurementRecord:
    """Remove rows where any numeric variable exceeds *zscore_threshold*."""

    numeric = _numeric_variables(record.data)
    if not numeric:
        return record
    dim = _sample_dim(record.data)
    length = int(record.data.sizes.get(dim, 0))
    if length == 0:
        return record
    mask = np.zeros(length, dtype=bool)
    for name in numeric:
        values = record.data[name].values.astype(float)
        std = values.std(ddof=0)
        if std == 0:
            continue
        zscores = (values - values.mean()) / std
        mask |= np.abs(zscores) > zscore_threshold
    if not mask.any():
        return record
    kept_indices = np.flatnonzero(~mask)
    cleaned = record.data.isel({dim: kept_indices})
    return replace(record, data=cleaned)


def rolling_average(record: MeasurementRecord, window: int = 5) -> MeasurementRecord:
    """Apply a rolling average to numeric variables to reduce noise."""

    if window <= 1:
        return record
    dim = _sample_dim(record.data)
    smoothed = record.data.rolling({dim: window}, min_periods=1).mean()
    smoothed = smoothed.fillna(record.data)
    return replace(record, data=smoothed)


def merge_records(records: Iterable[MeasurementRecord]) -> MeasurementRecord:
    """Concatenate multiple records that share metadata."""

    records = list(records)
    if not records:
        raise ValueError("At least one record is required for merging")
    base_meta = records[0].metadata
    for record in records[1:]:
        if record.metadata != base_meta:
            raise ValueError("All records must share identical metadata for merging")
    dim = _sample_dim(records[0].data)
    aligned = []
    offset = 0
    for record in records:
        length = int(record.data.sizes.get(dim, 0))
        coords = {dim: np.arange(offset, offset + length, dtype=int)}
        aligned.append(record.data.assign_coords(coords))
        offset += length
    concatenated = xr.concat(aligned, dim=dim)
    annotations = tuple(note for record in records for note in record.annotations)
    return MeasurementRecord(metadata=base_meta, data=concatenated, annotations=annotations)


def _dataset_from_columns(columns: dict[str, np.ndarray]) -> xr.Dataset:
    prepared = {name: np.atleast_1d(np.asarray(values, dtype=float)) for name, values in columns.items()}
    lengths = {len(values) for values in prepared.values()}
    if len(lengths) != 1:
        raise ValueError("All columns must share the same length")
    length = lengths.pop() if lengths else 0
    coords = {_SAMPLE_DIM: np.arange(length, dtype=int)}
    data_vars = {name: (_SAMPLE_DIM, values) for name, values in prepared.items()}
    dataset = xr.Dataset(data_vars=data_vars, coords=coords)
    dataset.attrs["data_columns"] = list(prepared)
    return dataset


def _structured_text_to_dataset(path: str | Path, *, delimiter: str = ",") -> xr.Dataset:
    array = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=float, encoding="utf-8")
    if array.size == 0:
        return xr.Dataset()
    names = array.dtype.names or ()
    if not names:
        raise ValueError("File must include a header row to name columns")
    columns = {name.strip().lower(): array[name] for name in names}
    return _dataset_from_columns(columns)


class CSVVoltammogramReader(BaseFileReader):
    """Parse generic CSV files exported from potentiostats."""

    technique = "echem"

    def read(self, path: str | Path, **kwargs: Any) -> MeasurementRecord:  # noqa: ARG002 - placeholder for future args
        dataset = _structured_text_to_dataset(path, delimiter=",")
        required = {"potential", "current"}
        missing = required - set(dataset.data_vars)
        if missing:
            raise ValueError(f"Electrochemistry file is missing columns: {sorted(missing)}")
        metadata = MeasurementMetadata(sample_id=Path(path).stem, technique=self.technique)
        return MeasurementRecord(metadata=metadata, data=dataset)


class XPSBindingEnergyReader(BaseFileReader):
    """Parse tab-delimited binding energy exports."""

    technique = "xps"

    def read(
        self,
        path: str | Path,
        *,
        binding_column: str = "bindingenergy",
        intensity_column: str = "intensity",
        delimiter: str = "\t",
        **_: Any,
    ) -> MeasurementRecord:
        dataset = _structured_text_to_dataset(path, delimiter=delimiter)
        normalized = {name.lower(): name for name in dataset.data_vars}
        binding_key = normalized.get(binding_column.lower())
        intensity_key = normalized.get(intensity_column.lower())
        missing = [label for label, key in [("binding", binding_key), ("intensity", intensity_key)] if key is None]
        if missing:
            raise ValueError(f"XPS file missing required columns: {missing}")
        renamed = dataset.rename({binding_key: "binding_energy", intensity_key: "intensity"})
        metadata = MeasurementMetadata(sample_id=Path(path).stem, technique=self.technique)
        return MeasurementRecord(metadata=metadata, data=renamed)


def register_builtin_readers() -> None:
    """Register the built-in reader implementations."""

    from ..reading.registry import register_reader

    register_reader(CSVVoltammogramReader)
    register_reader(XPSBindingEnergyReader)


__all__ = [
    "CSVVoltammogramReader",
    "DataObject",
    "MeasurementMetadata",
    "MeasurementRecord",
    "XPSBindingEnergyReader",
    "drop_outliers",
    "merge_records",
    "register_builtin_readers",
    "rolling_average",
]
