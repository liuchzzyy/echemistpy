"""Utilities for persisting Measurement and Results objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from echemistpy.io.plugin_manager import get_plugin_manager
from echemistpy.io.structures import (
    AnalysisResult,
    AnalysisResultInfo,
    Measurement,
    MeasurementInfo,
)


def save_measurement(
    measurement: Measurement,
    info: MeasurementInfo,
    path: str | Path,
    *,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Persist a Measurement and its Info to disk using plugin system.

    Args:
        measurement: The Measurement object to save.
        info: The MeasurementInfo object.
        path: Destination path.
        fmt: Optional format override (csv, nc, h5, json).
        **kwargs: Additional arguments for the saver.
    """
    destination = Path(path)

    # Convert MeasurementInfo to metadata dict
    metadata = info.to_dict()

    # Use plugin manager to save
    pm = get_plugin_manager()
    pm.save_data(measurement.data, metadata, destination, fmt=fmt, **kwargs)


def save_results(
    results: AnalysisResult,
    results_info: AnalysisResultInfo,
    path: str | Path,
    measurement: Optional[Measurement] = None,
    measurement_info: Optional[MeasurementInfo] = None,
    *,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Persist Results (and optionally original Measurement) to disk using plugin system.

    Args:
        results: The Results object to save.
        results_info: The ResultsInfo object.
        path: Destination path.
        measurement: Optional original Measurement to include.
        measurement_info: Optional original MeasurementInfo to include.
        fmt: Optional format override.
        **kwargs: Additional arguments.
    """
    destination = Path(path)
    extension = (fmt or destination.suffix.lstrip(".")).lower()

    # Convert info to metadata dict
    results_metadata = results_info.to_dict()

    # For HDF5/NetCDF formats, we can save multiple groups
    if extension in {"h5", "hdf5", "hdf", "nc", "nc4", "netcdf"}:
        import json

        # If measurement is provided, save it first
        if measurement and measurement_info:
            meas_dataset = measurement.data.copy()
            # Sanitize variable names
            rename_dict = {name: name.replace("/", "_") for name in meas_dataset.data_vars if "/" in name}
            if rename_dict:
                meas_dataset = meas_dataset.rename(rename_dict)

            meas_metadata = measurement_info.to_dict()
            meas_dataset.attrs["echemistpy_metadata"] = json.dumps(meas_metadata)
            meas_dataset.to_netcdf(destination, group="measurement", mode="w", engine="h5netcdf")
            mode = "a"
        else:
            mode = "w"

        # Save results
        res_dataset = results.data.copy()
        rename_dict = {name: name.replace("/", "_") for name in res_dataset.data_vars if "/" in name}
        if rename_dict:
            res_dataset = res_dataset.rename(rename_dict)

        res_dataset.attrs["echemistpy_results_metadata"] = json.dumps(results_metadata)
        res_dataset.to_netcdf(destination, group="results", mode=mode, engine="h5netcdf")
    else:
        # For other formats, just save results
        pm = get_plugin_manager()
        pm.save_data(results.data, results_metadata, destination, fmt=fmt, **kwargs)


__all__ = [
    "save_measurement",
    "save_results",
]
