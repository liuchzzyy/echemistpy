"""Generic file format saver plugins."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr
from traitlets import HasTraits, Unicode

from echemistpy.io.plugin_specs import hookimpl

logger = logging.getLogger(__name__)


class CSVSaverPlugin(HasTraits):
    """CSV/TSV file saver plugin."""

    delimiter = Unicode(default_value=",", help="Delimiter character").tag(config=True)
    encoding = Unicode(default_value="utf-8", help="File encoding").tag(config=True)

    @hookimpl
    def get_supported_formats(self) -> list[str]:
        """Return list of supported output formats."""
        return ["csv", "tsv", "txt"]

    @hookimpl
    def save_data(
        self,
        data: xr.Dataset,
        metadata: dict[str, Any],
        filepath: Path,
        fmt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save data to CSV/TSV file.

        Args:
            data: xarray.Dataset to save
            metadata: Metadata dictionary
            filepath: Destination path
            fmt: Optional format override
            **kwargs: Additional parameters
        """
        # Determine delimiter
        if fmt == "tsv" or filepath.suffix.lower() == ".tsv":
            delimiter = kwargs.get("delimiter", "\t")
        elif fmt == "txt" or filepath.suffix.lower() == ".txt":
            delimiter = kwargs.get("delimiter", "\t")
        else:
            delimiter = kwargs.get("delimiter", self.delimiter)

        encoding = kwargs.get("encoding", self.encoding)
        newline = kwargs.get("newline", "\n")

        # Select row dimension
        row_dim = self._select_row_dim(data)
        fieldnames, records = self._dataset_records(data, row_dim)

        with open(filepath, "w", newline=newline, encoding=encoding) as handle:
            # Write metadata header if provided
            if metadata:
                handle.write(f"# Metadata: {json.dumps(metadata)}\n")
                if "technique" in metadata:
                    handle.write(f"# Technique: {metadata['technique']}\n")
                if "sample_name" in metadata:
                    handle.write(f"# Sample: {metadata['sample_name']}\n")

            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    @staticmethod
    def _select_row_dim(dataset: xr.Dataset) -> str:
        """Select the row dimension."""
        if "row" in dataset.dims:
            return "row"
        if len(dataset.dims) == 1:
            return next(iter(dataset.dims))
        raise ValueError("CSV export requires a single dimension or an explicit 'row' dimension.")

    @staticmethod
    def _coerce_scalar(value: Any) -> Any:
        """Convert numpy scalars to Python types."""
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _dataset_records(dataset: xr.Dataset, row_dim: str) -> tuple[Sequence[str], list[dict[str, Any]]]:
        """Convert dataset to records for CSV writing."""
        size = dataset.dims.get(row_dim, 0)
        var_names = list(dataset.data_vars)
        records: list[dict[str, Any]] = []

        for idx in range(size):
            record: dict[str, Any] = {}
            for name in var_names:
                array = dataset[name]
                if array.ndim == 0:
                    record[name] = CSVSaverPlugin._coerce_scalar(array.values)
                elif array.dims == (row_dim,):
                    record[name] = CSVSaverPlugin._coerce_scalar(array.values[idx])
                else:
                    raise ValueError(
                        f"Variable '{name}' is not 1D aligned with '{row_dim}'. Cannot export to CSV."
                    )
            records.append(record)

        return var_names, records


class HDF5SaverPlugin(HasTraits):
    """HDF5/NetCDF file saver plugin."""

    engine = Unicode(default_value="h5netcdf", help="xarray engine").tag(config=True)

    @hookimpl
    def get_supported_formats(self) -> list[str]:
        """Return list of supported output formats."""
        return ["h5", "hdf5", "hdf", "nc", "nc4", "netcdf"]

    @hookimpl
    def save_data(
        self,
        data: xr.Dataset,
        metadata: dict[str, Any],
        filepath: Path,
        fmt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save data to HDF5/NetCDF file.

        Args:
            data: xarray.Dataset to save
            metadata: Metadata dictionary
            filepath: Destination path
            fmt: Optional format override
            **kwargs: Additional parameters
        """
        engine = kwargs.get("engine", self.engine)

        # Make a copy to avoid modifying original
        dataset = data.copy()

        # Sanitize variable names (replace / with _)
        rename_dict = {name: name.replace("/", "_") for name in dataset.data_vars if "/" in name}
        if rename_dict:
            dataset = dataset.rename(rename_dict)

        # Add metadata as attributes
        dataset.attrs["echemistpy_metadata"] = json.dumps(metadata)

        if "technique" in metadata:
            dataset.attrs["technique"] = str(metadata["technique"])
        if "sample_name" in metadata:
            dataset.attrs["sample_name"] = str(metadata["sample_name"])

        # Add other metadata fields
        for k, v in metadata.items():
            if k not in dataset.attrs and k not in ["technique", "sample_name"]:
                try:
                    dataset.attrs[k] = v
                except Exception:
                    dataset.attrs[k] = str(v)

        # Save to file
        dataset.to_netcdf(filepath, engine=engine, **kwargs)


class JSONSaverPlugin(HasTraits):
    """JSON file saver plugin."""

    indent = Unicode(default_value="2", help="JSON indentation").tag(config=True)
    encoding = Unicode(default_value="utf-8", help="File encoding").tag(config=True)

    @hookimpl
    def get_supported_formats(self) -> list[str]:
        """Return list of supported output formats."""
        return ["json"]

    @hookimpl
    def save_data(
        self,
        data: xr.Dataset,
        metadata: dict[str, Any],
        filepath: Path,
        fmt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save data to JSON file.

        Args:
            data: xarray.Dataset to save
            metadata: Metadata dictionary
            filepath: Destination path
            fmt: Optional format override
            **kwargs: Additional parameters
        """
        indent_val = int(kwargs.get("indent", self.indent))
        encoding = kwargs.get("encoding", self.encoding)

        # Convert dataset to dictionary
        output = {
            "metadata": metadata,
            "data": {},
        }

        for var_name in data.data_vars:
            var_data = data[var_name].values
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(var_data, np.ndarray):
                var_data = var_data.tolist()
            output["data"][var_name] = var_data

        # Save to file
        with open(filepath, "w", encoding=encoding) as f:
            json.dump(output, f, indent=indent_val, ensure_ascii=False)


__all__ = [
    "CSVSaverPlugin",
    "HDF5SaverPlugin",
    "JSONSaverPlugin",
]
