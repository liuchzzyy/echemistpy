"""Tests for NeXus template helpers."""

import xarray as xr

from echemistpy.io.structures import create_nxxbase_template, NXField


def test_nxfield_to_dataarray_attrs():
    field = NXField(name="temperature", value=300.0, dtype="NX_FLOAT", units="K", doc="Sample temp")
    data = field.to_dataarray()
    assert isinstance(data, xr.DataArray)
    assert data.attrs["type"] == "NX_FLOAT"
    assert data.attrs["units"] == "K"
    assert data.attrs["EX_doc"] == "Sample temp"


def test_nxxbase_template_to_xarray_tree_contains_key_groups():
    template = create_nxxbase_template()
    tree = template.to_xarray_tree()

    assert "entry" in tree
    assert "entry/instrument/detector" in tree
    assert "entry/data" in tree

    entry = tree["entry"]
    assert entry.attrs["NX_class"] == "NXentry"
    assert "title" in entry
    assert entry["title"].attrs["type"] == "NX_CHAR"

    detector = tree["entry/instrument/detector"]
    assert detector["data"].attrs["EX_doc"].startswith("The area detector data")

    data_group = tree["entry/data"]
    links = data_group.attrs["NX_links"]
    assert ("data", "/entry/instrument/detector/data", (("signal", "1"),)) in links
