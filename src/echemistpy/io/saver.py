"""Utilities for persisting Measurement and Results objects."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import xarray as xr

from echemistpy.io.structures import Measurement, MeasurementInfo, AnalysisResult, AnalysisResultInfo

_TEXT_FORMATS = {"csv": ","}
_NETCDF_FORMATS = {"nc", "nc4", "netcdf", "h5", "hdf5", "hdf"}


def _select_row_dim(dataset: xr.Dataset) -> str:
    if "row" in dataset.dims:
        return "row"
    if len(dataset.dims) == 1:
        return next(iter(dataset.dims))
    raise ValueError(
        "Text exports require a single dimension or an explicit 'row' dimension."
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
                # Skip non-1D variables for text export or raise error?
                # User said "Check if data is 1D/2D". If 2D (e.g. images), we can't easily save to CSV.
                # But here we are iterating rows, so we assume tabular data (1D arrays sharing a dim).
                # If we have true 2D data (matrix), this loop logic won't work well.
                raise ValueError(
                    f"Variable '{name}' is not 1D aligned with '{row_dim}'. Cannot export to CSV."
                )
        records.append(record)
    return var_names, records


def _write_text(
    path: Path,
    dataset: xr.Dataset,
    info: Optional[MeasurementInfo] = None,
    *,
    delimiter: str,
    newline: str = "\n",
) -> None:
    row_dim = _select_row_dim(dataset)
    fieldnames, records = _dataset_records(dataset, row_dim)
    
    with path.open("w", newline=newline, encoding="utf-8") as handle:
        # Write metadata header if provided
        if info:
            meta_dict = info.to_dict()
            # Write as JSON-like comments
            handle.write(f"# Metadata: {json.dumps(meta_dict)}\n")
            # Or write specific fields
            handle.write(f"# Technique: {info.technique}\n")
            handle.write(f"# Sample: {info.sample_name}\n")
            
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def save_measurement(
    measurement: Measurement,
    info: MeasurementInfo,
    path: str | Path,
    *,
    fmt: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Persist a Measurement and its Info to disk.

    Args:
        measurement: The Measurement object to save.
        info: The MeasurementInfo object.
        path: Destination path.
        fmt: Optional format override (csv, nc, h5).
        **kwargs: Additional arguments for the saver.
    """
    destination = Path(path)
    extension = (fmt or destination.suffix.lstrip(".")).lower()

    if extension in _TEXT_FORMATS:
        # Check for 1D/2D compatibility
        # We assume if it has a 'row' dim and variables are 1D along it, it's tabular (1D).
        # If variables are 2D, we can't save to single CSV easily.
        is_tabular = True
        try:
            _select_row_dim(measurement.data)
            # Also check variables
            for var in measurement.data.data_vars:
                if measurement.data[var].ndim > 1:
                    is_tabular = False
                    break
        except ValueError:
            is_tabular = False
            
        if not is_tabular:
            raise ValueError("Data is not 1D tabular. Cannot save to CSV.")

        delimiter = _TEXT_FORMATS[extension]
        _write_text(destination, measurement.data, info, delimiter=delimiter)
        return

    if extension in _NETCDF_FORMATS:
        dataset = measurement.data.copy()
        
        # Sanitize variable names (replace / with _)
        rename_dict = {name: name.replace("/", "_") for name in dataset.data_vars if "/" in name}
        if rename_dict:
            dataset = dataset.rename(rename_dict)
        
        # Save info as attributes
        # We serialize the whole info dict as a JSON string attribute for safety/completeness
        # and also set individual attributes for convenience.
        info_dict = info.to_dict()
        dataset.attrs["echemistpy_info"] = json.dumps(info_dict)
        
        dataset.attrs["technique"] = info.technique
        dataset.attrs["sample_name"] = info.sample_name
        if info.instrument:
            dataset.attrs["instrument"] = info.instrument
        if info.operator:
            dataset.attrs["operator"] = info.operator
            
        # Flatten extras into attributes if possible
        for k, v in info.extras.items():
            if k not in dataset.attrs:
                try:
                    # Try to save as is (if supported by backend) or stringify
                    dataset.attrs[k] = v
                except Exception:
                    dataset.attrs[k] = str(v)

        # Use h5netcdf for all supported formats if available, as it's lighter and pure python
        engine = "h5netcdf"
        dataset.to_netcdf(destination, engine=engine, **kwargs)
        return

    raise ValueError(f"Unsupported file extension '{extension}'. Only CSV and HDF5/NetCDF are supported.")


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
    """Persist Results (and optionally original Measurement) to disk.

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

    if extension in _TEXT_FORMATS:
        # For CSV, we save the results data.
        # If measurement is provided, we can't easily save both in one CSV.
        # We'll just save the results.
        
        # Check tabular
        try:
            _select_row_dim(results.data)
        except ValueError:
             raise ValueError("Results data is not 1D tabular. Cannot save to CSV.")

        delimiter = _TEXT_FORMATS[extension]
        # We can create a combined info object or just pass results_info
        # Let's pass results_info but maybe add a note about original measurement
        _write_text(destination, results.data, None, delimiter=delimiter) # TODO: Handle info for results in CSV
        return

    if extension in _NETCDF_FORMATS:
        engine = "h5netcdf"
        
        # If saving everything, we use groups.
        # Root group can be the Measurement (if present) or Results.
        
        mode = "w"
        if destination.exists():
            destination.unlink()
            
        if measurement and measurement_info:
            # Save Measurement as root or group "measurement"
            # Let's save as group "measurement" to be clean
            ds_meas = measurement.data.copy()
            # Sanitize
            rename_dict = {name: name.replace("/", "_") for name in ds_meas.data_vars if "/" in name}
            if rename_dict:
                ds_meas = ds_meas.rename(rename_dict)
                
            ds_meas.attrs["echemistpy_info"] = json.dumps(measurement_info.to_dict())
            ds_meas.to_netcdf(destination, group="measurement", mode="w", engine=engine)
            mode = "a"
            
        # Save Results as group "results"
        ds_res = results.data.copy()
        # Sanitize
        rename_dict = {name: name.replace("/", "_") for name in ds_res.data_vars if "/" in name}
        if rename_dict:
            ds_res = ds_res.rename(rename_dict)
            
        ds_res.attrs["echemistpy_results_info"] = json.dumps(results_info.to_dict())
        ds_res.to_netcdf(destination, group="results", mode=mode, engine=engine)
        
        return

    raise ValueError(f"Unsupported file extension '{extension}'. Only CSV and HDF5/NetCDF are supported.")
