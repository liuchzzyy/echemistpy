"""Unified file saving interface for scientific data.

This module provides a simplified interface for saving data.
It supports common formats like CSV, JSON, and NetCDF/HDF5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import xarray as xr

from echemistpy.io.structures import (
    RawData,
    ResultsData,
)


def save_dataset(
    dataset: xr.Dataset,
    path: str | Path,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Save an xarray.Dataset to disk.

    Args:
        dataset: The xarray.Dataset to save
        path: Destination path
        fmt: Optional format override ('csv', 'nc', 'h5', 'json')
        **kwargs: Additional arguments for the specific saver
    """
    path = Path(path)
    ext = fmt.lower() if fmt else path.suffix.lower().lstrip(".")

    if ext in {"csv", "txt", "tsv"}:
        # Convert to pandas and save
        sep = "\t" if ext == "tsv" else ","
        df = dataset.to_dataframe()
        # If it's a simple 1D dataset with 'record' or 'row' dimension, we might want to reset index
        if "record" in df.index.names:
            df = df.reset_index(drop=True)
        elif "row" in df.index.names:
            df = df.reset_index(drop=True)
        df.to_csv(path, sep=sep, index=False, **kwargs)

    elif ext in {"nc", "netcdf", "h5", "hdf5"}:
        # Save as NetCDF (standard for xarray)
        dataset.to_netcdf(path, **kwargs)

    elif ext == "json":
        # Save as JSON (useful for metadata or small datasets)
        data_dict = dataset.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=False)

    else:
        raise ValueError(f"Unsupported save format: {ext}")


def save_raw_data(
    raw_data: RawData,
    path: str | Path,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Save a RawData object to disk.

    Args:
        raw_data: The RawData object to save
        path: Destination path
        fmt: Optional format override
    """
    # We save the underlying xarray dataset
    # Metadata is typically stored in dataset.attrs or separately
    save_dataset(raw_data.data, path, fmt=fmt, **kwargs)


def save_results_data(
    results_data: ResultsData,
    path: str | Path,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Save a ResultsData object to disk.

    Args:
        results_data: The ResultsData object to save
        path: Destination path
        fmt: Optional format override
    """
    save_dataset(results_data.data, path, fmt=fmt, **kwargs)


__all__ = [
    "save_dataset",
    "save_raw_data",
    "save_results_data",
]
