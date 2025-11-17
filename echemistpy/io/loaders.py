"""Data loaders for various file formats."""

from pathlib import Path
from typing import Any

import h5py
import pandas as pd
import xarray as xr

from echemistpy.core.exceptions import DataLoadError


def load_excel(
    filepath: Path | str,
    sheet_name: str | int | None = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load data from Excel file.

    Parameters
    ----------
    filepath : Path or str
        Path to Excel file
    sheet_name : str, int, or None, optional
        Sheet name or index to load, by default 0
    **kwargs : Any
        Additional arguments passed to pd.read_excel

    Returns
    -------
    pd.DataFrame
        Loaded data

    Raises
    ------
    DataLoadError
        If file cannot be loaded
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"File not found: {filepath}"
            raise DataLoadError(msg)

        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    except Exception as e:
        msg = f"Failed to load Excel file {filepath}: {e}"
        raise DataLoadError(msg) from e


def load_csv(
    filepath: Path | str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load data from CSV file.

    Parameters
    ----------
    filepath : Path or str
        Path to CSV file
    **kwargs : Any
        Additional arguments passed to pd.read_csv

    Returns
    -------
    pd.DataFrame
        Loaded data

    Raises
    ------
    DataLoadError
        If file cannot be loaded
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"File not found: {filepath}"
            raise DataLoadError(msg)

        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        msg = f"Failed to load CSV file {filepath}: {e}"
        raise DataLoadError(msg) from e


def load_hdf5(
    filepath: Path | str,
    key: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame | dict[str, Any]:
    """Load data from HDF5 file.

    Parameters
    ----------
    filepath : Path or str
        Path to HDF5 file
    key : str, optional
        Key to load from HDF5 file. If None, returns all keys
    **kwargs : Any
        Additional arguments passed to pd.read_hdf or h5py.File

    Returns
    -------
    pd.DataFrame or dict
        Loaded data or dictionary of all datasets

    Raises
    ------
    DataLoadError
        If file cannot be loaded
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"File not found: {filepath}"
            raise DataLoadError(msg)

        if key is not None:
            # Load specific key using pandas
            return pd.read_hdf(filepath, key=key, **kwargs)

        # Load all keys using h5py
        data_dict = {}
        with h5py.File(filepath, "r") as f:

            def _load_dataset(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    data_dict[name] = obj[()]

            f.visititems(_load_dataset)
        return data_dict
    except Exception as e:
        msg = f"Failed to load HDF5 file {filepath}: {e}"
        raise DataLoadError(msg) from e


def load_netcdf(
    filepath: Path | str,
    **kwargs: Any,
) -> xr.Dataset:
    """Load data from NetCDF file.

    Parameters
    ----------
    filepath : Path or str
        Path to NetCDF file
    **kwargs : Any
        Additional arguments passed to xr.open_dataset

    Returns
    -------
    xr.Dataset
        Loaded dataset

    Raises
    ------
    DataLoadError
        If file cannot be loaded
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"File not found: {filepath}"
            raise DataLoadError(msg)

        return xr.open_dataset(filepath, **kwargs)
    except Exception as e:
        msg = f"Failed to load NetCDF file {filepath}: {e}"
        raise DataLoadError(msg) from e
