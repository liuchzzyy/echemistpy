"""Tests for NeXus template helpers."""

import xarray as xr

from echemistpy.io.structures import NXEchemBase, NXField, create_nxxbase_template


def test_nxfield_to_dataarray_attrs():
    field = NXField(name="temperature", value=300.0, dtype="NX_FLOAT", units="K", doc="Sample temp")
    data = field.to_dataarray()
    assert isinstance(data, xr.DataArray)
    assert data.attrs["type"] == "NX_FLOAT"
    assert data.attrs["units"] == "K"
    assert data.attrs["EX_doc"] == "Sample temp"


def test_nxechembase_produces_expected_tree():
    template = NXEchemBase(title="Custom Title", readme="Notes").to_nxfile()
    tree = template.to_xarray_tree()

    assert "entry" in tree
    assert "entry/RawData/Echem" in tree
    assert "entry/RawData/Echem/MetaData" in tree
    assert "entry/Results/Echem" in tree

    entry = tree["entry"]
    assert entry.attrs["NX_class"] == "NXentry"
    assert str(entry["Title"].item()) == "Custom Title"
    assert str(entry["Readme"].item()) == "Notes"

    raw_meta = tree["entry/RawData/Echem/MetaData"]
    assert "InstrumentInfo" in raw_meta
    assert raw_meta["InstrumentInfo"].attrs["type"] == "NX_CHAR"
    assert "Time" in raw_meta

    results_echem = tree["entry/Results/Echem"]
    assert results_echem["Data"].attrs["type"] == "NX_FLOAT"


def test_create_nxxbase_template_keeps_public_api():
    template = create_nxxbase_template()
    tree = template.to_xarray_tree()

    assert "entry/RawData/Echem" in tree
