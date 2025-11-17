"""Utilities for persisting :class:`xarray.Dataset` objects."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import xarray as xr

_TEXT_FORMATS = {"csv": ",", "tsv": "\t"}
_JSON_FORMATS = {"json"}
_NETCDF_FORMATS = {"nc", "nc4", "netcdf"}


def _select_row_dim(dataset: xr.Dataset) -> str:
    if "row" in dataset.dims:
        return "row"
    if len(dataset.dims) == 1:
        return next(iter(dataset.dims))
    raise ValueError(
        "Text and JSON exports require a single dimension or an explicit 'row' dimension."
    )


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset_records(dataset: xr.Dataset, row_dim: str) -> tuple[Sequence[str], list[dict[str, Any]]]:
    size = dataset.dims.get(row_dim, 0)
    var_names = list(dataset.data_vars)
    records: list[dict[str, Any]] = []
    for idx in range(size):
        record: dict[str, Any] = {}
        for name in var_names:
            array = dataset[name]
            if array.ndim == 0:
                record[name] = _coerce_scalar(array.values)
            elif array.dims == (row_dim,):
                record[name] = _coerce_scalar(array.values[idx])
            else:
                raise ValueError(
                    "Only 1D variables aligned with the primary dimension can be exported to text/JSON."
                )
        records.append(record)
    return var_names, records


def _write_text(
    path: Path,
    dataset: xr.Dataset,
    *,
    delimiter: str,
    newline: str = "\n",
) -> None:
    row_dim = _select_row_dim(dataset)
    fieldnames, records = _dataset_records(dataset, row_dim)
    with path.open("w", newline=newline, encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _write_json(path: Path, dataset: xr.Dataset, *, indent: int = 2) -> None:
    row_dim = _select_row_dim(dataset)
    _, records = _dataset_records(dataset, row_dim)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=indent)


def save_table(
    dataset: xr.Dataset,
    path: str | Path,
    *,
    fmt: Optional[str] = None,
    storage_options: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """Persist a dataset to disk using simple tabular formats."""

    del storage_options  # Reserved for future remote backends

    destination = Path(path)
    extension = (fmt or destination.suffix.lstrip(".")).lower()

    if extension in _TEXT_FORMATS:
        delimiter = _TEXT_FORMATS[extension]
        _write_text(destination, dataset, delimiter=delimiter)
        return

    if extension in _JSON_FORMATS:
        indent = kwargs.pop("indent", 2)
        _write_json(destination, dataset, indent=indent)
        return

    if extension in _NETCDF_FORMATS:
        dataset.to_netcdf(destination, **kwargs)
        return

    raise ValueError(f"Unsupported file extension '{extension}'.")
