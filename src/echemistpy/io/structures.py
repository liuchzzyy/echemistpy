"""Shared data structures for both :mod:`xarray` and NeXus containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Tuple

import xarray as xr


@dataclass(slots=True)
class Axis:
    """Describe a measurement axis and its sampling."""

    name: str
    unit: Optional[str] = None
    values: Optional[Sequence[float]] = None


@dataclass(slots=True)
class MeasurementMetadata:
    """Holds descriptive metadata for a measurement."""

    technique: str
    sample_name: str
    instrument: Optional[str] = None
    operator: Optional[str] = None
    extras: MutableMapping[str, Any] = field(default_factory=dict)

    def copy(self) -> "MeasurementMetadata":
        return MeasurementMetadata(
            technique=self.technique,
            sample_name=self.sample_name,
            instrument=self.instrument,
            operator=self.operator,
            extras=dict(self.extras),
        )


@dataclass(slots=True)
class Measurement:
    """Container representing the raw data from an experiment."""

    data: xr.Dataset
    metadata: MeasurementMetadata
    axes: List[Axis] = field(default_factory=list)

    def copy(self) -> "Measurement":
        return Measurement(
            data=self.data.copy(deep=True),
            metadata=self.metadata.copy(),
            axes=[Axis(axis.name, axis.unit, axis.values) for axis in self.axes],
        )

    def require_variables(self, variables: Iterable[str]) -> None:
        missing = [name for name in variables if name not in self.data.variables]
        if missing:
            raise ValueError("Measurement is missing required variables: " + ", ".join(missing))

    # Backwards compatible alias
    def require_columns(self, columns: Iterable[str]) -> None:  # pragma: no cover - thin wrapper
        self.require_variables(columns)


@dataclass(slots=True)
class AnalysisResult:
    """A light-weight container for processed data."""

    technique: str
    sample_name: str
    summary: Dict[str, Any]
    tables: Dict[str, xr.Dataset] = field(default_factory=dict)
    figures: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "AnalysisResult") -> "AnalysisResult":
        if self.sample_name != other.sample_name:
            raise ValueError("Cannot merge results from different samples.")
        merged_summary = {**self.summary, **other.summary}
        merged_tables = {**self.tables, **other.tables}
        merged_figures = {**self.figures, **other.figures}
        return AnalysisResult(
            technique=self.technique,
            sample_name=self.sample_name,
            summary=merged_summary,
            tables=merged_tables,
            figures=merged_figures,
        )


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
    doc: Optional[str] = None
    extra_attrs: MutableMapping[str, Any] = field(default_factory=dict)

    def to_dataarray(self) -> xr.DataArray:
        """Represent the field as a scalar :class:`xarray.DataArray`."""

        data = xr.DataArray(self.value)
        data.attrs["type"] = self.dtype
        data.attrs["EX_required"] = "true" if self.required else "false"
        if self.units:
            data.attrs["units"] = self.units
        if self.doc:
            data.attrs["EX_doc"] = self.doc
        for key, value in self.extra_attrs.items():
            data.attrs[key] = value
        return data


@dataclass(slots=True)
class NXLink:
    """Representation of a NeXus soft link."""

    name: str
    target: str
    extra_attrs: MutableMapping[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> Tuple[str, str, Tuple[Tuple[str, Any], ...]]:
        """Return a tuple representation suitable for dataset attributes."""

        extras = tuple(sorted(self.extra_attrs.items()))
        return (self.name, self.target, extras)


@dataclass(slots=True)
class NXGroup:
    """Representation of a NeXus group with nested content."""

    name: str
    nx_class: str
    required: bool = True
    doc: Optional[str] = None
    attributes: MutableMapping[str, Any] = field(default_factory=dict)
    fields: List[NXField] = field(default_factory=list)
    groups: List["NXGroup"] = field(default_factory=list)
    links: List[NXLink] = field(default_factory=list)

    def to_dataset(self) -> xr.Dataset:
        """Represent the group as an :class:`xarray.Dataset`."""

        data_vars = {field.name: field.to_dataarray() for field in self.fields}
        dataset = xr.Dataset(data_vars=data_vars)
        dataset.attrs["NX_class"] = self.nx_class
        dataset.attrs["EX_required"] = "true" if self.required else "false"
        if self.doc:
            dataset.attrs["EX_doc"] = self.doc
        for key, value in self.attributes.items():
            dataset.attrs[key] = value
        if self.links:
            dataset.attrs["NX_links"] = tuple(link.to_tuple() for link in self.links)
        return dataset

    def iter_datasets(self, parent_path: str = "") -> Iterator[Tuple[str, xr.Dataset]]:
        """Yield ``(path, dataset)`` pairs for this group and its children."""

        path = "/".join(part for part in (parent_path, self.name) if part)
        yield path, self.to_dataset()
        for subgroup in self.groups:
            yield from subgroup.iter_datasets(path)


@dataclass(slots=True)
class NXFile:
    """Container for describing NeXus file layouts via :mod:`xarray`."""

    groups: List[NXGroup] = field(default_factory=list)
    attrs: MutableMapping[str, Any] = field(default_factory=dict)

    def iter_datasets(self) -> Iterator[Tuple[str, xr.Dataset]]:
        """Iterate over every ``NXGroup`` as ``(path, dataset)`` pairs."""

        for group in self.groups:
            yield from group.iter_datasets("")

    def to_xarray_tree(self) -> Dict[str, xr.Dataset]:
        """Materialize the full hierarchy as ``path -> Dataset`` mapping."""

        return dict(self.iter_datasets())


@dataclass(slots=True)
class NXEchemBase:
    """Utility that builds the electrochemistry-oriented NXxbase layout."""

    title: str = "Project Title"
    readme: str = "Describe the experiment"

    @staticmethod
    def _nx_field(
        name: str,
        value: Any,
        dtype: str,
        *,
        units: Optional[str] = None,
        doc: Optional[str] = None,
        required: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> NXField:
        return NXField(
            name=name,
            value=value,
            dtype=dtype,
            required=required,
            units=units,
            doc=doc,
            extra_attrs=extra or {},
        )

    def _raw_echem_group(self) -> NXGroup:
        raw_echem_metadata = NXGroup(
            name="MetaData",
            nx_class="NXcollection",
            fields=[
                self._nx_field(
                    "InstrumentInfo",
                    "Describe potentiostat, reference, etc.",
                    "NX_CHAR",
                    doc="Basic instrument description",
                ),
                self._nx_field(
                    "Measurement",
                    "Detail electrolyte, scan rate, etc.",
                    "NX_CHAR",
                    doc="Acquisition settings",
                ),
                self._nx_field(
                    "Time",
                    "2024-01-01T00:00:00",
                    "NX_DATE_TIME",
                    doc="Acquisition timestamp",
                ),
            ],
        )

        return NXGroup(
            name="Echem",
            nx_class="NXcollection",
            fields=[
                self._nx_field("Title", "Electrochemistry", "NX_CHAR"),
                self._nx_field(
                    "Data",
                    0.0,
                    "NX_FLOAT",
                    doc="Placeholder for raw electrochemistry data table",
                ),
            ],
            groups=[raw_echem_metadata],
        )

    def _results_echem_group(self) -> NXGroup:
        results_echem_metadata = NXGroup(
            name="MetaData",
            nx_class="NXcollection",
            fields=[
                self._nx_field(
                    "Analysis",
                    "Describe processing pipeline and parameters",
                    "NX_CHAR",
                    doc="Analysis provenance",
                )
            ],
        )

        return NXGroup(
            name="Echem",
            nx_class="NXcollection",
            fields=[
                self._nx_field(
                    "Data",
                    0.0,
                    "NX_FLOAT",
                    doc="Placeholder for processed electrochemistry metrics",
                )
            ],
            groups=[results_echem_metadata],
        )

    def to_nxfile(self) -> NXFile:
        """Materialize the NXxbase layout anchored around electrochemistry."""

        entry = NXGroup(
            name="entry",
            nx_class="NXentry",
            attributes={"default": "Results"},
            fields=[
                self._nx_field("Title", self.title, "NX_CHAR"),
                self._nx_field("Readme", self.readme, "NX_CHAR"),
            ],
            groups=[
                NXGroup(
                    name="RawData",
                    nx_class="NXcollection",
                    doc="Container for technique-specific raw datasets (Echem, XRD, TEM, …)",
                    groups=[self._raw_echem_group()],
                ),
                NXGroup(
                    name="Results",
                    nx_class="NXcollection",
                    doc="Derived analysis products per technique",
                    groups=[self._results_echem_group()],
                ),
            ],
        )

        return NXFile(groups=[entry], attrs={"default": "entry"})


def create_nxxbase_template() -> NXFile:
    """Backward compatible helper that returns :class:`NXEchemBase`."""

    return NXEchemBase().to_nxfile()


__all__ = [
    "AnalysisResult",
    "Axis",
    "Measurement",
    "MeasurementMetadata",
    "NXEchemBase",
    "NXField",
    "NXFile",
    "NXGroup",
    "NXLink",
    "create_nxxbase_template",
]
