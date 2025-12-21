"""Generic file format loader plugins."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from traitlets import HasTraits, Unicode

from echemistpy.io.plugin_specs import hookimpl
from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class CSVLoaderPlugin(HasTraits):
    """CSV/TSV file loader plugin."""

    delimiter = Unicode(default_value=",", help="Delimiter character").tag(config=True)
    encoding = Unicode(default_value="utf-8", help="File encoding").tag(config=True)

    @hookimpl
    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return ["csv", "txt", "tsv"]

    @hookimpl
    def can_load(self, filepath: Path) -> bool:
        """Check if this loader can handle the given file."""
        return filepath.suffix.lower() in [".csv", ".txt", ".tsv"]

    @hookimpl
    def load_file(
        self,
        filepath: Path,
        **kwargs: Any,
    ) -> tuple[RawData, RawDataInfo]:
        """Load a delimited text file.

        Args:
            filepath: Path to the file
            **kwargs: Additional parameters (delimiter, encoding, etc.)

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        # Determine delimiter based on extension
        if filepath.suffix.lower() == ".tsv":
            delimiter = kwargs.get("delimiter", "\t")
        elif filepath.suffix.lower() == ".txt":
            delimiter = kwargs.get("delimiter", "\t")
        else:
            delimiter = kwargs.get("delimiter", self.delimiter)

        encoding = kwargs.get("encoding", self.encoding)

        # Read header lines to capture potential metadata
        header_lines = []
        with open(filepath, "r", encoding=encoding, errors="ignore") as f:
            for _ in range(10):  # Read first 10 lines to check for comments
                line = f.readline()
                if line.strip().startswith("#") or line.strip().startswith("%"):
                    header_lines.append(line.strip())
                else:
                    break

        # Load data using numpy
        array = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            names=True,
            dtype=None,
            encoding=encoding,
        )

        # Convert to dataset
        dataset = self._array_to_dataset(array)

        # Create metadata
        meta = {
            "technique": "Table",
            "filename": filepath.stem,
            "extension": filepath.suffix,
            "delimiter": delimiter,
        }
        if header_lines:
            meta["header_comments"] = header_lines

        raw_data = RawData(data=dataset)
        raw_data_info = RawDataInfo(meta=meta)

        return raw_data, raw_data_info

    @staticmethod
    def _array_to_dataset(array: np.ndarray) -> xr.Dataset:
        """Convert structured numpy array to xarray Dataset."""
        if array.size == 0:
            return xr.Dataset()
        if array.ndim == 0:
            array = array.reshape(1)
        if array.dtype.names is None:
            raise ValueError("Tabular text files must include a header row.")

        row_dim = np.arange(array.shape[0])
        data_vars = {name: ("row", array[name]) for name in array.dtype.names}
        return xr.Dataset(data_vars=data_vars, coords={"row": row_dim})


class ExcelLoaderPlugin(HasTraits):
    """Excel file loader plugin."""

    sheet_name = Unicode(default_value="0", help="Sheet name or index").tag(config=True)

    @hookimpl
    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return ["xlsx", "xls"]

    @hookimpl
    def can_load(self, filepath: Path) -> bool:
        """Check if this loader can handle the given file."""
        # This is a generic Excel loader - lower priority than LANHE
        return filepath.suffix.lower() in [".xlsx", ".xls"]

    @hookimpl
    def load_file(
        self,
        filepath: Path,
        **kwargs: Any,
    ) -> tuple[RawData, RawDataInfo]:
        """Load an Excel file.

        Args:
            filepath: Path to the file
            **kwargs: Additional parameters (sheet_name, etc.)

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        sheet_name = kwargs.get("sheet_name", 0)

        # Read Excel file
        df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

        # Convert to xarray Dataset
        data_vars = {}
        for col in df.columns:
            data_vars[str(col)] = ("row", df[col].values)

        dataset = xr.Dataset(data_vars=data_vars, coords={"row": np.arange(len(df))})

        meta = {
            "technique": "Excel",
            "filename": filepath.stem,
            "extension": filepath.suffix,
            "sheet_name": sheet_name,
        }

        raw_data = RawData(data=dataset)
        raw_data_info = RawDataInfo(meta=meta)

        return raw_data, raw_data_info


class HDF5LoaderPlugin(HasTraits):
    """HDF5/NetCDF file loader plugin."""

    engine = Unicode(default_value="h5netcdf", help="xarray engine").tag(config=True)

    @hookimpl
    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return ["h5", "hdf5", "hdf", "nc", "nc4", "netcdf"]

    @hookimpl
    def can_load(self, filepath: Path) -> bool:
        """Check if this loader can handle the given file."""
        return filepath.suffix.lower() in [".h5", ".hdf5", ".hdf", ".nc", ".nc4", ".netcdf"]

    @hookimpl
    def load_file(
        self,
        filepath: Path,
        **kwargs: Any,
    ) -> tuple[RawData, RawDataInfo]:
        """Load an HDF5/NetCDF file.

        Args:
            filepath: Path to the file
            **kwargs: Additional parameters (engine, etc.)

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        engine = kwargs.get("engine", self.engine)

        try:
            dataset = xr.open_dataset(filepath, engine=engine, **kwargs)
        except Exception:
            # Fallback to netcdf4 engine
            dataset = xr.open_dataset(filepath, engine="netcdf4", **kwargs)

        # Extract metadata from dataset attributes
        meta = dict(dataset.attrs)
        meta.update({
            "technique": meta.get("technique", "HDF5"),
            "filename": filepath.stem,
            "extension": filepath.suffix,
        })

        raw_data = RawData(data=dataset)
        raw_data_info = RawDataInfo(meta=meta)

        return raw_data, raw_data_info


__all__ = [
    "CSVLoaderPlugin",
    "ExcelLoaderPlugin",
    "HDF5LoaderPlugin",
]
