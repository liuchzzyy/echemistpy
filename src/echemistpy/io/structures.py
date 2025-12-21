"""Shared data structures for both :mod:`xarray` and NeXus containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional

import pandas as pd
import xarray as xr


class MetadataInfoMixin:
    """Mixin providing common metadata operations for Info classes.

    This mixin provides shared functionality for RawDataInfo, MeasurementInfo,
    and AnalysisResultInfo classes.
    """

    STANDARD_METADATA = {
        "technique": "Unknown",
        "sample_name": "Unknown",
        "start_time": None,
        "end_time": None,
        "instrument": None,
        "operator": None,
    }

    def __post_init__(self):
        """Initialize standard metadata in 'others' if present."""
        if hasattr(self, "others") and isinstance(self.others, dict):
            for key, value in self.STANDARD_METADATA.items():
                if key not in self.others:
                    self.others[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation.

        Returns:
            Dictionary containing all metadata fields
        """
        from dataclasses import asdict

        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key.

        Checks standard fields first, then dynamic storage (meta, parameters, others).

        Args:
            key: Metadata key to retrieve
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        # Check standard fields first
        if hasattr(self, key):
            return getattr(self, key)

        # Check dynamic storage locations in order of precedence
        # RawDataInfo -> meta
        # AnalysisResultInfo -> parameters
        # MeasurementInfo -> others
        for container_name in ["meta", "parameters", "others"]:
            container = getattr(self, container_name, None)
            if isinstance(container, dict) and key in container:
                return container[key]

        return default

    def update(self, other: dict[str, Any]) -> None:
        """Update metadata with new key-value pairs.

        Updates standard fields if they exist, otherwise updates the appropriate
        dynamic storage (meta, parameters, or others).

        Args:
            other: Dictionary of metadata to add/update
        """
        from dataclasses import fields

        try:
            field_names = {f.name for f in fields(self)}
        except TypeError:
            field_names = set()

        # Determine dynamic container
        container = None
        if hasattr(self, "meta"):
            container = self.meta
        elif hasattr(self, "parameters"):
            container = self.parameters
        elif hasattr(self, "others"):
            container = self.others

        for key, value in other.items():
            if key in field_names and key not in {"meta", "parameters", "others"}:
                setattr(self, key, value)
            elif container is not None:
                container[key] = value


class XarrayDataMixin:
    """Mixin providing common xarray.Dataset operations.

    This mixin eliminates code duplication across RawData, Measurement,
    and AnalysisResult classes by providing shared functionality.
    """

    data: xr.Dataset  # Type hint for IDE support

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

    def get_variables(self) -> list[str]:
        """Get list of all variable names in the dataset.

        Returns:
            List of variable names
        """
        return list(self.data.data_vars.keys())

    def get_coords(self) -> list[str]:
        """Get list of all coordinate names in the dataset.

        Returns:
            List of coordinate names
        """
        return list(self.data.coords.keys())

    @cached_property
    def variables(self) -> list[str]:
        """Get list of all variable names (cached).

        Returns:
            List of variable names
        """
        return list(self.data.data_vars.keys())

    @cached_property
    def coords(self) -> list[str]:
        """Get list of all coordinate names (cached).

        Returns:
            List of coordinate names
        """
        return list(self.data.coords.keys())

    def select(self, variables: Optional[list[str]] = None) -> xr.Dataset:
        """Select specific variables from the dataset.

        Args:
            variables: List of variable names to select, or None for all

        Returns:
            xarray.Dataset with selected variables
        """
        if variables is None:
            return self.data
        return self.data[variables]

    def __getitem__(self, key: str) -> xr.DataArray:
        """Access a variable by name.

        Args:
            key: Variable name

        Returns:
            xarray.DataArray for the variable
        """
        return self.data[key]


@dataclass(slots=True)
class RawDataInfo(MetadataInfoMixin):
    """Container for all metadata extracted from the raw file.

    This stores unprocessed metadata directly from file loaders, maintaining
    the original keys and values from the source file.

    Attributes:
        meta: Dictionary containing all metadata key-value pairs
    """

    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation.

        Returns:
            Dictionary containing all metadata
        """
        return self.meta


@dataclass(slots=True)
class RawData(XarrayDataMixin):
    """Container for raw data using xarray.Dataset as the underlying structure.

    This represents unprocessed data directly from file loaders, maintaining
    the original structure and column names from the source file. The xarray
    backend provides powerful data manipulation and analysis capabilities.

    Attributes:
        data: xarray.Dataset containing the raw measurement data
    """

    data: xr.Dataset


@dataclass(slots=True)
class MeasurementInfo(MetadataInfoMixin):
    """Standardized metadata for a measurement.

    This class provides a consistent interface for measurement metadata across
    different file formats and techniques. It separates common fields from
    technique-specific metadata stored in the 'others' dictionary.
    """

    others: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Measurement(XarrayDataMixin):
    """Container for standardized measurement data using xarray.Dataset.

    This represents measurement data after standardization, with consistent
    column names and units following echemistpy conventions. The xarray
    backend enables powerful data analysis and visualization capabilities.

    Metadata is stored separately in MeasurementInfo for clean separation
    of data and metadata.

    Attributes:
        data: xarray.Dataset containing standardized measurement data
    """

    data: xr.Dataset


@dataclass(slots=True)
class AnalysisResultInfo(MetadataInfoMixin):
    """Metadata for analysis results.

    This class stores metadata about data analysis and processing, including
    parameters used, remarks about the analysis, and any additional metadata.

    Attributes:
        parameters: Dictionary of analysis parameters and settings
        others: Additional metadata not covered by standard fields
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    others: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisResult(XarrayDataMixin):
    """Container for processed analysis results using xarray.Dataset.

    This represents data after analysis and processing, such as fitted parameters,
    derived metrics, or processed signals. The xarray backend provides powerful
    data manipulation and visualization capabilities.

    Metadata about the analysis is stored separately in AnalysisResultInfo for clean
    separation of data and metadata.

    Attributes:
        data: xarray.Dataset containing analysis results
    """

    data: xr.Dataset


__all__ = [
    "AnalysisResult",
    "AnalysisResultInfo",
    "Measurement",
    "MeasurementInfo",
    "RawData",
    "RawDataInfo",
]
