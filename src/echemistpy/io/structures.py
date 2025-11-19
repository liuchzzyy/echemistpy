"""Shared data structures for both :mod:`xarray` and NeXus containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterator, MutableMapping, Optional

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
            if key in field_names and key not in ["meta", "parameters", "others"]:
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


@dataclass(slots=True)
class NXField:
    """Representation of a NeXus dataset.

    The layout mirrors the generated example in
    ``ex_h5py_NXxbase.py`` where each dataset stores its NX data type,
    requirement flag, optional units, and documentation string.
    """

    name: str
    value: Any
    dtype: str
    required: bool = True
    units: Optional[str] = None
    extra_attrs: MutableMapping[str, Any] = field(default_factory=dict)

    def to_dataarray(self) -> xr.DataArray:
        """Represent the field as a scalar :class:`xarray.DataArray`."""

        data = xr.DataArray(self.value)
        data.attrs["type"] = self.dtype
        data.attrs["EX_required"] = "true" if self.required else "false"
        if self.units:
            data.attrs["units"] = self.units
        for key, value in self.extra_attrs.items():
            data.attrs[key] = value
        return data
    
    @classmethod
    def from_dataarray(
        cls,
        name: str,
        dataarray: xr.DataArray,
        dtype: Optional[str] = None,
    ) -> "NXField":
        """Create NXField from an xarray.DataArray.
        
        Args:
            name: Field name
            dataarray: xarray.DataArray to convert
            dtype: Optional NX data type override
            
        Returns:
            NXField instance
        """
        # Extract value (use first element if array)
        value = dataarray.values
        if hasattr(value, 'item'):
            try:
                value = value.item()
            except (ValueError, AttributeError):
                pass  # Keep as array if multi-element
        
        # Extract attributes
        attrs = dict(dataarray.attrs)
        units = attrs.pop("units", None)
        nx_dtype = dtype or attrs.pop("type", "NX_FLOAT")
        required = attrs.pop("EX_required", "true") == "true"
        
        return cls(
            name=name,
            value=value,
            dtype=nx_dtype,
            required=required,
            units=units,
            extra_attrs=attrs,
        )


@dataclass(slots=True)
class NXLink:
    """Representation of a NeXus soft link."""

    name: str
    target: str
    extra_attrs: MutableMapping[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> tuple[str, str, tuple[tuple[str, Any], ...]]:
        """Return a tuple representation suitable for dataset attributes."""

        extras = tuple(sorted(self.extra_attrs.items()))
        return (self.name, self.target, extras)


@dataclass(slots=True)
class NXGroup:
    """Representation of a NeXus group with nested content."""

    name: str
    nx_class: str
    required: bool = True
    attributes: MutableMapping[str, Any] = field(default_factory=dict)
    fields: list[NXField] = field(default_factory=list)
    groups: list["NXGroup"] = field(default_factory=list)
    links: list[NXLink] = field(default_factory=list)

    def to_dataset(self) -> xr.Dataset:
        """Represent the group as an :class:`xarray.Dataset`."""

        data_vars = {field.name: field.to_dataarray() for field in self.fields}
        dataset = xr.Dataset(data_vars=data_vars)
        dataset.attrs["NX_class"] = self.nx_class
        dataset.attrs["EX_required"] = "true" if self.required else "false"
        for key, value in self.attributes.items():
            dataset.attrs[key] = value
        if self.links:
            dataset.attrs["NX_links"] = tuple(link.to_tuple() for link in self.links)
        return dataset

    def iter_datasets(self, parent_path: str = "") -> Iterator[tuple[str, xr.Dataset]]:
        """Yield ``(path, dataset)`` pairs for this group and its children."""

        path = f"{parent_path}/{self.name}" if parent_path else self.name
        yield path, self.to_dataset()
        for subgroup in self.groups:
            yield from subgroup.iter_datasets(path)
    
    def add_field(self, field: NXField) -> "NXGroup":
        """Add a field to this group (fluent interface).
        
        Args:
            field: NXField to add
            
        Returns:
            Self for method chaining
        """
        self.fields.append(field)
        return self
    
    def add_group(self, group: "NXGroup") -> "NXGroup":
        """Add a subgroup to this group (fluent interface).
        
        Args:
            group: NXGroup to add as child
            
        Returns:
            Self for method chaining
        """
        self.groups.append(group)
        return self
    
    def add_link(self, link: NXLink) -> "NXGroup":
        """Add a link to this group (fluent interface).
        
        Args:
            link: NXLink to add
            
        Returns:
            Self for method chaining
        """
        self.links.append(link)
        return self
    
    @classmethod
    def from_dataset(
        cls,
        name: str,
        dataset: xr.Dataset,
        nx_class: str = "NXcollection",
    ) -> "NXGroup":
        """Create NXGroup from an xarray.Dataset.
        
        Args:
            name: Group name
            dataset: xarray.Dataset to convert
            nx_class: NeXus class name
            
        Returns:
            NXGroup instance
        """
        # Extract attributes
        attrs = dict(dataset.attrs)
        attrs.pop("EX_doc", None)  # Remove doc if present
        required = attrs.pop("EX_required", "true") == "true"
        attrs.pop("NX_class", None)  # Remove NX_class from attributes
        
        # Convert data variables to fields
        fields = []
        for var_name in dataset.data_vars:
            field = NXField.from_dataarray(var_name, dataset[var_name])
            fields.append(field)
        
        return cls(
            name=name,
            nx_class=nx_class,
            required=required,
            attributes=attrs,
            fields=fields,
        )


@dataclass(slots=True)
class NXFile:
    """Container for describing NeXus file layouts via :mod:`xarray`."""

    groups: list[NXGroup] = field(default_factory=list)
    attrs: MutableMapping[str, Any] = field(default_factory=dict)

    def iter_datasets(self) -> Iterator[tuple[str, xr.Dataset]]:
        """Iterate over every ``NXGroup`` as ``(path, dataset)`` pairs."""

        for group in self.groups:
            yield from group.iter_datasets("")

    def to_xarray_tree(self) -> dict[str, xr.Dataset]:
        """Materialize the full hierarchy as ``path -> Dataset`` mapping."""

        return dict(self.iter_datasets())

@dataclass
class Entry:
    """Container for a single electrochemical entry (test).
    
    Holds the six standard data and info objects defined in structures.py.
    """
    raw_data: RawData
    raw_data_info: RawDataInfo
    measurement: Measurement
    measurement_info: MeasurementInfo
    analysis_result: AnalysisResult
    analysis_result_info: AnalysisResultInfo

    def to_nx_group(self, name: str = "entry") -> NXGroup:
        """Convert this entry to a NeXus group (NXentry).
        
        Args:
            name: Name of the group (default: "entry")
            
        Returns:
            NXGroup representing this entry
        """
        entry_group = NXGroup(name=name, nx_class="NXentry")

        # Helper to convert data/info pair to NXGroup
        def _add_subgroup(
            data_obj: Union[RawData, Measurement, AnalysisResult],
            info_obj: Union[RawDataInfo, MeasurementInfo, AnalysisResultInfo],
            group_name: str,
            nx_class: str
        ) -> None:
            if data_obj and data_obj.data is not None:
                # Create group from dataset
                # Note: from_dataset extracts attributes from the dataset
                subgroup = NXGroup.from_dataset(
                    name=group_name,
                    dataset=data_obj.data,
                    nx_class=nx_class
                )
                
                # Add info metadata as attributes to the group
                if info_obj:
                    subgroup.attributes.update(info_obj.to_dict())
                
                entry_group.add_group(subgroup)

        # Add the three main data components as subgroups
        _add_subgroup(self.raw_data, self.raw_data_info, "raw_data", "NXdata")
        _add_subgroup(self.measurement, self.measurement_info, "measurement", "NXdata")
        _add_subgroup(self.analysis_result, self.analysis_result_info, "analysis_result", "NXdata")

        return entry_group


@dataclass
class Entries:
    """Container for electrochemical tests.
    
    Can hold one or multiple entries.
    """
    entries: List[Entry] = field(default_factory=list)

    def add_entry(self, entry: Entry) -> None:
        """Add an entry to the test."""
        self.entries.append(entry)

    @property
    def container(self) -> Union[Entry, NXFile]:
        """Build the appropriate container based on the number of entries.
        
        Returns:
            Entry: If there is exactly one entry (direct access to 6 categories).
            NXFile: If there are multiple entries (organized as NeXus file).
        """
        if len(self.entries) == 1:
            # Case 1: Single entry -> Return the entry object directly
            return self.entries[0]
        else:
            # Case 2: Multiple entries -> Organize as NXFile
            nx_file = NXFile()
            for i, entry in enumerate(self.entries):
                # Name entries as entry1, entry2, etc.
                group_name = f"entry{i+1}"
                nx_file.groups.append(entry.to_nx_group(name=group_name))
            return nx_file

__all__ = [
    # Base data types (six fundamental types)
    "RawData",
    "RawDataInfo",
    "Measurement",
    "MeasurementInfo",
    "AnalysisResult",
    "AnalysisResultInfo",
    # NX structures
    "Entry",
    "Entries",
]
