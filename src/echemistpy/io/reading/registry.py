"""Lightweight plugin registry for file readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Type

import numpy as np
import xarray as xr

from ..reorganization import MeasurementMetadata, MeasurementRecord
from .base import BaseFileReader


class ReaderNotRegisteredError(KeyError):
    """Raised when attempting to use a technique without a registered reader."""


_registry: Dict[str, Type[BaseFileReader]] = {}


def register_reader(reader_cls: Type[BaseFileReader]) -> Type[BaseFileReader]:
    """Register *reader_cls* for its declared technique."""

    technique = reader_cls.technique
    if not technique:
        msg = "Technique readers must define a non-empty 'technique' attribute"
        raise ValueError(msg)
    _registry[technique.lower()] = reader_cls
    return reader_cls


def registry_snapshot() -> Dict[str, str]:
    """Return a read-only view of the registered techniques."""

    return {technique: reader.__name__ for technique, reader in _registry.items()}


def get_reader(technique: str) -> BaseFileReader:
    """Instantiate the reader that handles *technique*."""

    reader_cls = _registry.get(technique.lower())
    if reader_cls is None:
        raise ReaderNotRegisteredError(
            f"No reader registered for technique '{technique}'. "
            "Use `register_reader` to make one available."
        )
    return reader_cls()


def load_measurement(
    path: str | Path,
    technique: str,
    *,
    sample_id: str | None = None,
    instrument: str | None = None,
    operator: str | None = None,
    metadata_extra: dict | None = None,
    **reader_kwargs: Any,
) -> MeasurementRecord:
    """Helper that reads a measurement file and attaches metadata."""

    reader = get_reader(technique)
    record = reader(path, **reader_kwargs)
    if sample_id or instrument or operator or metadata_extra:
        metadata = MeasurementMetadata(
            sample_id=sample_id or record.metadata.sample_id,
            technique=record.metadata.technique,
            instrument=instrument or record.metadata.instrument,
            operator=operator or record.metadata.operator,
            run_date=record.metadata.run_date,
            extra={**record.metadata.extra, **(metadata_extra or {})},
        )
        record = MeasurementRecord(metadata=metadata, data=record.data, annotations=record.annotations)
    return record


def dataset_from_records(records: Iterable[MeasurementRecord]) -> xr.Dataset:
    """Combine multiple :class:`MeasurementRecord` objects into an :class:`xarray.Dataset`."""

    flattened: list[dict[str, Any]] = []
    for record in records:
        summary = record.summary()
        summary["metadata"] = record.metadata.describe()
        flat = _flatten_summary(summary)
        flattened.append(flat)
    if not flattened:
        return xr.Dataset()
    keys = sorted({key for flat in flattened for key in flat})
    coords = {"record": np.arange(len(flattened), dtype=int)}
    data_vars: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for key in keys:
        values = [flat.get(key) for flat in flattened]
        if all((value is None) or isinstance(value, (int, float, np.number)) for value in values):
            array = np.array([np.nan if value is None else float(value) for value in values], dtype=float)
        else:
            array = np.array(values, dtype=object)
        data_vars[key] = (("record",), array)
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


__all__ = [
    "BaseFileReader",
    "ReaderNotRegisteredError",
    "dataset_from_records",
    "get_reader",
    "load_measurement",
    "register_reader",
    "registry_snapshot",
]
