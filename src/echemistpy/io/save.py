"""Utilities to persist :class:`echemistpy.io.reorganization.DataObject` instances."""

from __future__ import annotations

import json
from pathlib import Path

import xarray as xr

from .reorganization import DataObject


def save_data_object(
    data_object: DataObject,
    path: str | Path,
    *,
    format: str = "netcdf",
    engine: str = "h5netcdf",
    **kwargs,
) -> Path:
    """Serialize *data_object* to disk via :mod:`xarray` backends."""

    dataset = data_object.to_dataset()
    metadata_attr = dataset.attrs.get("metadata")
    if isinstance(metadata_attr, dict):
        dataset.attrs["metadata"] = json.dumps(metadata_attr, default=str)
    output_path = Path(path)

    if format.lower() == "netcdf":
        dataset.to_netcdf(output_path, engine=engine, **kwargs)
    elif format.lower() == "zarr":
        dataset.to_zarr(output_path, mode="w", **kwargs)
    else:  # pragma: no cover - defensive branch
        msg = "Unsupported format. Use 'netcdf' or 'zarr'."
        raise ValueError(msg)

    return output_path


def dataset_from_records(records: list[DataObject]) -> xr.Dataset:
    """Concatenate multiple data objects along a new sample dimension."""

    labeled = []
    for idx, obj in enumerate(records):
        dataset = obj.to_dataset()
        dataset = dataset.expand_dims({"sample": [idx]})
        labeled.append(dataset)
    if not labeled:
        return xr.Dataset()
    return xr.concat(labeled, dim="sample")


__all__ = ["save_data_object", "dataset_from_records"]
