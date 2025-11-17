"""Shared data structures for both :mod:`xarray` and NeXus containers."""

from __future__ import annotations

import datetime as _datetime
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

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
            raise ValueError(
                "Measurement is missing required variables: " + ", ".join(missing)
            )

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


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive import helper
        raise RuntimeError("h5py is required to serialize NeXus structures") from exc
    return h5py


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

    def to_h5py(self, parent: Any) -> Any:
        """Materialize the field inside an ``h5py`` group."""

        h5py = _require_h5py()
        dataset = parent.create_dataset(name=self.name, data=self.value, maxshape=None)
        dataset.attrs["type"] = self.dtype
        dataset.attrs["EX_required"] = "true" if self.required else "false"
        if self.units:
            dataset.attrs["units"] = self.units
        if self.doc:
            dataset.attrs["EX_doc"] = self.doc
        for key, value in self.extra_attrs.items():
            dataset.attrs[key] = value
        return dataset


@dataclass(slots=True)
class NXLink:
    """Representation of a NeXus soft link."""

    name: str
    target: str
    extra_attrs: MutableMapping[str, Any] = field(default_factory=dict)

    def to_h5py(self, parent: Any) -> Any:
        h5py = _require_h5py()
        parent[self.name] = h5py.SoftLink(self.target)
        link = parent[self.name]
        link.attrs["target"] = self.target
        for key, value in self.extra_attrs.items():
            link.attrs[key] = value
        return link


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

    def to_h5py(self, parent: Any) -> Any:
        """Write the group and its children into an ``h5py`` file."""

        group = parent.create_group(self.name)
        group.attrs["NX_class"] = self.nx_class
        group.attrs["EX_required"] = "true" if self.required else "false"
        if self.doc:
            group.attrs["EX_doc"] = self.doc
        for key, value in self.attributes.items():
            group.attrs[key] = value
        for field in self.fields:
            field.to_h5py(group)
        for subgroup in self.groups:
            subgroup.to_h5py(group)
        for link in self.links:
            link.to_h5py(group)
        return group


@dataclass(slots=True)
class NXFile:
    """Container for serializing NeXus file layouts."""

    groups: List[NXGroup] = field(default_factory=list)
    attrs: MutableMapping[str, Any] = field(default_factory=dict)

    def write(self, destination: Any) -> None:
        """Serialize the NeXus layout using :mod:`h5py`.

        ``destination`` may either be a filesystem path or an ``h5py.File``
        instance. File level metadata such as ``file_name`` and
        ``file_time`` follow the behaviour from
        ``ex_h5py_NXxbase.py``.
        """

        h5py = _require_h5py()

        manage_file = not isinstance(destination, h5py.File)
        handle = destination
        path_hint: Optional[str] = None
        if manage_file:
            path_hint = os.fspath(destination)
            handle = h5py.File(path_hint, "w")
        else:  # pragma: no cover - exercised in integration environments
            path_hint = getattr(destination, "filename", None)

        try:
            for group in self.groups:
                group.to_h5py(handle)
            for key, value in self.attrs.items():
                handle.attrs[key] = value

            if path_hint:
                handle.attrs.setdefault("file_name", os.path.abspath(path_hint))
            handle.attrs.setdefault("file_time", _datetime.datetime.now().isoformat())
            handle.attrs.setdefault("h5py_version", h5py.version.version)
            handle.attrs.setdefault("HDF5_Version", h5py.version.hdf5_version)
        finally:
            if manage_file:
                handle.close()


def create_nxxbase_template() -> NXFile:
    """Create the canonical ``NXxbase`` hierarchy.

    The values follow the auto-generated NeXus example from
    ``ex_h5py_NXxbase.py`` and provide a ready-to-serialize template for
    instrument control data.
    """

    def nx_field(
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

    source = NXGroup(
        name="source",
        nx_class="NXsource",
        fields=[
            nx_field("type", "SAMPLE-CHAR-DATA", "NX_CHAR"),
            nx_field("name", "SAMPLE-CHAR-DATA", "NX_CHAR"),
            nx_field("probe", "neutron", "NX_CHAR"),
        ],
    )

    monochromator = NXGroup(
        name="monochromator",
        nx_class="NXmonochromator",
        fields=[
            nx_field(
                "wavelength",
                1.0,
                "NX_FLOAT",
                units="NX_WAVELENGTH",
            )
        ],
    )

    detector = NXGroup(
        name="detector",
        nx_class="NXdetector",
        doc=(
            "The name of the group is detector if there is only one detector, "
            "if there are several, names have to be detector1, detector2, ...detectorn."
        ),
        fields=[
            nx_field(
                "data",
                1,
                "NX_INT",
                doc=(
                    "The area detector data, the first dimension is always the number of scan "
                    "points, the second and third are the number of pixels in x and y. The "
                    "origin is always assumed to be in the center of the detector. maxOccurs is "
                    "limited to the number of detectors on your instrument."
                ),
                extra={"signal": "1"},
            ),
            nx_field("x_pixel_size", 1.0, "NX_FLOAT", units="NX_LENGTH"),
            nx_field("y_pixel_size", 1.0, "NX_FLOAT", units="NX_LENGTH"),
            nx_field("distance", 1.0, "NX_FLOAT", units="NX_LENGTH"),
            nx_field(
                "frame_start_number",
                1,
                "NX_INT",
                doc=(
                    "This is the start number of the first frame of a scan. In PX one often "
                    "scans a couple of frames on a give sample, then does something else, then "
                    "returns to the same sample and scans some more frames. Each time with a new "
                    "data file. This number helps concatenating such measurements."
                ),
            ),
        ],
    )

    instrument = NXGroup(
        name="instrument",
        nx_class="NXinstrument",
        groups=[source, monochromator, detector],
    )

    sample = NXGroup(
        name="sample",
        nx_class="NXsample",
        fields=[
            nx_field("name", "SAMPLE-CHAR-DATA", "NX_CHAR", doc="Descriptive name of sample"),
            nx_field(
                "orientation_matrix",
                1.0,
                "NX_FLOAT",
                doc=(
                    "The orientation matrix according to Busing and Levy conventions. This is "
                    "not strictly necessary as the UB can always be derived from the data. But "
                    "let us bow to common usage which includes the UB nearly always."
                ),
            ),
            nx_field(
                "unit_cell",
                1.0,
                "NX_FLOAT",
                doc=(
                    "The unit cell, a, b, c, alpha, beta, gamma. Again, not strictly necessary, "
                    "but normally written."
                ),
            ),
            nx_field("temperature", 1.0, "NX_FLOAT", doc="The sample temperature or whatever sensor represents this value best"),
            nx_field(
                "x_translation",
                1.0,
                "NX_FLOAT",
                units="NX_LENGTH",
                doc="Translation of the sample along the X-direction of the laboratory coordinate system",
            ),
            nx_field(
                "y_translation",
                1.0,
                "NX_FLOAT",
                units="NX_LENGTH",
                doc="Translation of the sample along the Y-direction of the laboratory coordinate system",
            ),
            nx_field(
                "distance",
                1.0,
                "NX_FLOAT",
                units="NX_LENGTH",
                doc="Translation of the sample along the Z-direction of the laboratory coordinate system",
            ),
        ],
    )

    control = NXGroup(
        name="control",
        nx_class="NXmonitor",
        fields=[
            nx_field(
                "mode",
                "monitor",
                "NX_CHAR",
                doc="Count to a preset value based on either clock time (timer) or received monitor counts (monitor).",
            ),
            nx_field("preset", 1.0, "NX_FLOAT", doc="preset value for time or monitor"),
            nx_field(
                "integral",
                1.0,
                "NX_FLOAT",
                units="NX_ANY",
                doc="Total integral monitor counts",
            ),
        ],
    )

    data_group = NXGroup(
        name="data",
        nx_class="NXdata",
        doc=(
            "The name of this group id data if there is only one detector; if there are several the names "
            "will be data1, data2, data3 and will point to the corresponding detector groups in the instrument hierarchy."
        ),
        attributes={"signal": "data"},
        links=[
            NXLink(
                name="data",
                target="/entry/instrument/detector/data",
                extra_attrs={"signal": "1"},
            )
        ],
    )

    entry = NXGroup(
        name="entry",
        nx_class="NXentry",
        attributes={"default": "data"},
        fields=[
            nx_field("title", "SAMPLE-CHAR-DATA", "NX_CHAR"),
            nx_field(
                "start_time",
                "2021-03-29T15:51:46.136351",
                "NX_DATE_TIME",
            ),
            nx_field(
                "definition",
                "NXxbase",
                "NX_CHAR",
                doc="Official NeXus NXDL schema to which this file conforms",
            ),
        ],
        groups=[instrument, sample, control, data_group],
    )

    nx_file = NXFile(groups=[entry], attrs={"default": "entry"})
    return nx_file
