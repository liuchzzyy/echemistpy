"""Base classes for characterization techniques."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


class BaseData:
    """Base class for storing characterization data.

    This class provides a common interface for all data types, wrapping
    various data structures (numpy arrays, pandas DataFrames, xarray Datasets).

    Attributes
    ----------
    raw_data : Any
        The raw data in its original format
    metadata : dict
        Metadata associated with the data
    """

    def __init__(self, data: Any, metadata: dict | None = None):
        """Initialize BaseData.

        Parameters
        ----------
        data : Any
            Raw data (numpy array, pandas DataFrame, xarray Dataset, etc.)
        metadata : dict, optional
            Metadata dictionary
        """
        self.raw_data = data
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(data_type={type(self.raw_data).__name__})"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert data to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Data as DataFrame

        Raises
        ------
        NotImplementedError
            If conversion is not implemented for this data type
        """
        if isinstance(self.raw_data, pd.DataFrame):
            return self.raw_data
        if isinstance(self.raw_data, xr.Dataset):
            return self.raw_data.to_dataframe()
        if isinstance(self.raw_data, xr.DataArray):
            return self.raw_data.to_dataframe()
        if isinstance(self.raw_data, np.ndarray):
            return pd.DataFrame(self.raw_data)
        msg = f"Cannot convert {type(self.raw_data)} to DataFrame"
        raise NotImplementedError(msg)

    def to_xarray(self) -> xr.Dataset | xr.DataArray:
        """Convert data to xarray Dataset or DataArray.

        Returns
        -------
        xr.Dataset or xr.DataArray
            Data as xarray object

        Raises
        ------
        NotImplementedError
            If conversion is not implemented for this data type
        """
        if isinstance(self.raw_data, (xr.Dataset, xr.DataArray)):
            return self.raw_data
        if isinstance(self.raw_data, pd.DataFrame):
            return xr.Dataset.from_dataframe(self.raw_data)
        if isinstance(self.raw_data, np.ndarray):
            return xr.DataArray(self.raw_data)
        msg = f"Cannot convert {type(self.raw_data)} to xarray"
        raise NotImplementedError(msg)


class BaseCharacterization(ABC):
    """Abstract base class for all characterization techniques.

    This class defines the interface that all characterization technique
    classes should implement, ensuring consistent API across techniques.

    Attributes
    ----------
    data : BaseData
        The loaded data
    technique_name : str
        Name of the characterization technique
    """

    def __init__(self, technique_name: str):
        """Initialize BaseCharacterization.

        Parameters
        ----------
        technique_name : str
            Name of the characterization technique
        """
        self.technique_name = technique_name
        self.data: BaseData | None = None

    @abstractmethod
    def load_data(self, filepath: Path | str, **kwargs: Any) -> BaseData:
        """Load data from file.

        Parameters
        ----------
        filepath : Path or str
            Path to data file
        **kwargs : Any
            Additional keyword arguments for loading

        Returns
        -------
        BaseData
            Loaded data wrapped in BaseData

        Notes
        -----
        This method must be implemented by all subclasses.
        """

    @abstractmethod
    def preprocess(self, **kwargs: Any) -> BaseData:
        """Preprocess the loaded data.

        Parameters
        ----------
        **kwargs : Any
            Preprocessing parameters

        Returns
        -------
        BaseData
            Preprocessed data

        Notes
        -----
        This method must be implemented by all subclasses.
        """

    @abstractmethod
    def analyze(self, **kwargs: Any) -> dict[str, Any]:
        """Analyze the data.

        Parameters
        ----------
        **kwargs : Any
            Analysis parameters

        Returns
        -------
        dict
            Analysis results

        Notes
        -----
        This method must be implemented by all subclasses.
        """

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(technique={self.technique_name})"
