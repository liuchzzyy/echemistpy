"""Shared data structures for both :mod:`xarray` and NeXus containers."""

from __future__ import annotations

from typing import Any, Optional, cast

import pandas as pd
import xarray as xr
from traitlets import Dict, HasTraits, Instance, List, Unicode, Union


class MetadataInfoMixin:
    """Mixin providing common metadata operations for Info classes.

    This mixin provides shared functionality for RawDataInfo and ResultsDataInfo
    classes using traitlets.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation.

        Returns:
            Dictionary containing all metadata fields (excluding None values)
        """
        return {k: v for k, v in cast(HasTraits, self).trait_values().items() if v is not None}

    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key.

        Checks standard fields first, then dynamic storage (parameters, others).

        Args:
            key: Metadata key to retrieve
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        # Check standard fields first
        if hasattr(self, key) and cast(HasTraits, self).has_trait(key):
            return getattr(self, key)

        # Check dynamic storage locations in order of precedence
        for container_name in ["parameters", "others"]:
            container = getattr(self, container_name, None)
            if isinstance(container, dict) and key in container:
                return container[key]

        return default

    def update(self, other: dict[str, Any]) -> None:
        """Update metadata with new key-value pairs.

        Updates standard fields if they exist, otherwise updates the appropriate
        dynamic storage (parameters or others).

        Args:
            other: Dictionary of metadata to add/update
        """
        # Determine dynamic container
        container = None
        if hasattr(self, "parameters"):
            container = self.parameters
        elif hasattr(self, "others"):
            container = self.others

        for key, value in other.items():
            if cast(HasTraits, self).has_trait(key):
                setattr(self, key, value)
            elif container is not None and isinstance(container, dict):
                container[key] = value


class XarrayDataMixin:
    """Mixin providing common xarray.Dataset operations.

    This mixin eliminates code duplication across RawData and ResultsData
    classes by providing shared functionality.
    """

    data: xr.Dataset  # Type hint for IDE support

    def copy(self, deep: bool = True) -> Any:
        """Create a copy of the data object.

        Args:
            deep: Whether to perform a deep copy of the underlying dataset

        Returns:
            A new instance of the same class with copied data
        """
        return self.__class__(data=self.data.copy(deep=deep))

    def to_pandas(self) -> pd.DataFrame | pd.Series:
        """Convert xarray.Dataset to pandas DataFrame or Series.

        This method wraps xarray's to_pandas() which only works for Datasets
        with 1 or fewer dimensions. For multi-dimensional data, use to_dataframe().

        Returns:
            pandas.DataFrame for 1D data, pandas.Series for 0D data

        Raises:
            ValueError: If the Dataset has more than 1 dimension

        Note:
            - 0-dimensional Dataset → pandas.Series
            - 1-dimensional Dataset → pandas.DataFrame
            - For 2D+ data, use self.data.to_dataframe() instead
        """
        # Check number of dimensions
        n_dims = len(self.data.dims)

        if n_dims > 1:
            raise ValueError(
                f"to_pandas() only works for Datasets with 1 or fewer dimensions. "
                f"This Dataset has {n_dims} dimensions: {list(self.data.dims.keys())}. "
                f"Use self.data.to_dataframe() for multi-dimensional data."
            )

        return self.data.to_pandas()

    @property
    def variables(self) -> list[str]:
        """Get list of all variable names.

        Returns:
            List of variable names
        """
        return [str(k) for k in self.data.data_vars]

    @property
    def coords(self) -> list[str]:
        """Get list of all coordinate names.

        Returns:
            List of coordinate names
        """
        return [str(k) for k in self.data.coords]

    def get_variables(self) -> list[str]:
        """Get list of all variable names in the dataset."""
        return self.variables

    def get_coords(self) -> list[str]:
        """Get list of all coordinate names in the dataset."""
        return self.coords

    def select(self, variables: Optional[list[str]] = None) -> xr.Dataset:
        """Select specific variables from the dataset.

        Args:
            variables: List of variable names to select, or None for all

        Returns:
            xarray.Dataset with selected variables
        """
        if variables is None:
            return self.data
        # Ensure we return a Dataset even if one variable is selected
        result = self.data[variables]
        if isinstance(result, xr.DataArray):
            return result.to_dataset()
        return result

    def __getitem__(self, key: str) -> xr.DataArray:
        """Access a variable by name.

        Args:
            key: Variable name

        Returns:
            xarray.DataArray for the variable
        """
        return self.data[key]


class BaseInfo(HasTraits, MetadataInfoMixin):
    """Base class for all metadata info containers."""

    technique = List(Unicode(), default_value=["Unknown"])
    sample_name = Unicode("Unknown")
    start_time = Unicode(None, allow_none=True)
    operator = Unicode(None, allow_none=True)
    instrument = Unicode(None, allow_none=True)
    active_material_mass = Unicode(None, allow_none=True)
    wave_number = Unicode(None, allow_none=True)

    def copy(self) -> Any:
        """Create a copy of the info object.

        Returns:
            A new instance of the same class with copied metadata
        """
        return self.__class__(**self.to_dict())

    def __init__(self, **kwargs):
        """Initialize with standard metadata defaults.

        Args:
            **kwargs: Trait values or metadata to be stored in dynamic containers.
        """
        # Separate traits from other metadata
        trait_names = self.trait_names()
        traits = {k: v for k, v in kwargs.items() if k in trait_names}
        others_dict = {k: v for k, v in kwargs.items() if k not in trait_names}

        super().__init__(**traits)

        # Initialize dynamic storage if present
        for container_name in ["parameters", "others"]:
            if hasattr(self, container_name):
                container = getattr(self, container_name)
                if container is None:
                    setattr(self, container_name, {})
                    container = getattr(self, container_name)

                if isinstance(container, dict):
                    # Add extra kwargs that weren't traits
                    container.update(others_dict)


class BaseData(HasTraits, XarrayDataMixin):
    """Base class for all xarray-based data containers."""

    data = Union([Instance(xr.Dataset), Instance(xr.DataTree)], help="xarray.Dataset or xarray.DataTree containing the data")


class RawDataInfo(BaseInfo):
    """Container for all metadata extracted from the file.

    This stores metadata with standardized keys (technique, sample_name, etc.)
    while keeping original instrument-specific metadata in the 'others' dictionary.
    """

    others = Dict(help="Dictionary containing all metadata key-value pairs")


class RawData(BaseData):
    """Container for measurement data using xarray.Dataset.

    This represents the data loaded from a file, standardized with consistent
    column names and units.
    """

    pass


class ResultsDataInfo(BaseInfo):
    """Metadata for analysis results.

    This class stores metadata about data analysis and processing, including
    parameters used, remarks about the analysis, and any additional metadata.

    Attributes:
        parameters: Dictionary of analysis parameters and settings
        others: Additional metadata not covered by standard fields
    """

    parameters = Dict(help="Dictionary of analysis parameters and settings")
    others = Dict(help="Additional metadata not covered by standard fields")


class ResultsData(BaseData):
    """Container for processed results data using xarray.Dataset.

    This represents data after analysis and processing, such as fitted parameters,
    derived metrics, or processed signals. The xarray backend provides powerful
    data manipulation and visualization capabilities.

    Metadata about the analysis is stored separately in ResultsDataInfo for clean
    separation of data and metadata.
    """

    pass


__all__ = [
    "RawData",
    "RawDataInfo",
    "ResultsData",
    "ResultsDataInfo",
]
