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
    AnalysisResult,
    SummaryData,
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
        # If it's a simple 1D dataset with 'row' dimension, we might want to reset index
        if "row" in df.index.names:
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


def save_summary(
    summary_data: SummaryData,
    path: str | Path,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Save a SummaryData object to disk.

    Args:
        summary_data: The SummaryData object to save
        path: Destination path
        fmt: Optional format override
    """
    # We save the underlying xarray dataset
    # Metadata is typically stored in dataset.attrs or separately
    save_dataset(summary_data.data, path, fmt=fmt, **kwargs)


def save_analysis_result(
    result: AnalysisResult,
    path: str | Path,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Save an AnalysisResult object to disk.

    Args:
        result: The AnalysisResult object to save
        path: Destination path
        fmt: Optional format override
    """
    save_dataset(result.data, path, fmt=fmt, **kwargs)


__all__ = [
    "save_analysis_result",
    "save_dataset",
    "save_summary",
]
