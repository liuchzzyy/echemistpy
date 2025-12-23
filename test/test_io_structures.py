import pytest
import xarray as xr
import pandas as pd
import numpy as np
from echemistpy.io.structures import RawData, RawDataInfo, ResultsData, ResultsDataInfo


def test_raw_data_info():
    info = RawDataInfo(sample_name="TestSample", technique=["echem"], instrument="BioLogic")
    assert info.sample_name == "TestSample"
    assert info.technique == ["echem"]
    assert info.instrument == "BioLogic"

    # Test to_dict
    d = info.to_dict()
    assert d["sample_name"] == "TestSample"
    assert d["technique"] == ["echem"]

    # Test get
    assert info.get("sample_name") == "TestSample"
    assert info.get("non_existent", "default") == "default"

    # Test update
    info.update({"operator": "Alice", "custom_param": 123})
    assert info.operator == "Alice"
    assert info.others["custom_param"] == 123
    assert info.get("custom_param") == 123


def test_raw_data_info_copy():
    """Test the copy method of RawDataInfo."""
    info = RawDataInfo(sample_name="TestSample", technique=["echem"], instrument="BioLogic")
    info_copy = info.copy()
    assert info_copy.sample_name == info.sample_name
    assert info_copy is not info


def test_raw_data():
    ds = xr.Dataset({"current": (["record"], [1, 2, 3])}, coords={"record": [0, 1, 2]})
    rd = RawData(data=ds)
    assert rd.data.equals(ds)

    # Test copy
    rd_copy = rd.copy()
    assert rd_copy.data.equals(rd.data)
    assert rd_copy is not rd

    # Test to_pandas
    df = rd.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "current" in df.columns


def test_raw_data_variables_and_coords():
    """Test variables and coords properties."""
    ds = xr.Dataset(
        {"current": (["record"], [1, 2, 3]), "voltage": (["record"], [0.1, 0.2, 0.3])},
        coords={"record": [0, 1, 2], "time": (["record"], [0, 1, 2])},
    )
    rd = RawData(data=ds)

    # Test variables property
    assert "current" in rd.variables
    assert "voltage" in rd.variables
    assert len(rd.variables) == 2

    # Test coords property
    assert "record" in rd.coords
    assert "time" in rd.coords

    # Test get_variables and get_coords methods
    assert set(rd.get_variables()) == {"current", "voltage"}
    assert "record" in rd.get_coords()


def test_raw_data_select():
    """Test the select method."""
    ds = xr.Dataset(
        {"current": (["record"], [1, 2, 3]), "voltage": (["record"], [0.1, 0.2, 0.3])},
        coords={"record": [0, 1, 2]},
    )
    rd = RawData(data=ds)

    # Select specific variables
    selected = rd.select(["current"])
    assert "current" in selected.data_vars
    assert "voltage" not in selected.data_vars

    # Select all variables
    all_vars = rd.select(None)
    assert all_vars.equals(rd.data)


def test_raw_data_getitem():
    """Test the __getitem__ method."""
    ds = xr.Dataset({"current": (["record"], [1, 2, 3])}, coords={"record": [0, 1, 2]})
    rd = RawData(data=ds)

    current_array = rd["current"]
    assert isinstance(current_array, xr.DataArray)
    assert len(current_array) == 3


def test_raw_data_to_pandas_multidim_error():
    """Test that to_pandas raises error for multi-dimensional data."""
    ds = xr.Dataset({"data": (["x", "y"], [[1, 2], [3, 4]])}, coords={"x": [0, 1], "y": [0, 1]})
    rd = RawData(data=ds)

    with pytest.raises(ValueError, match="to_pandas.*only works for Datasets with 1 or fewer dimensions"):
        rd.to_pandas()


def test_results_data_info():
    info = ResultsDataInfo(technique=["echem"])
    info.parameters = {"analysis_type": "CV"}  # analysis_type is in parameters
    assert info.technique == ["echem"]
    assert info.parameters["analysis_type"] == "CV"

    d = info.to_dict()
    assert "parameters" in d
    assert d["parameters"]["analysis_type"] == "CV"


def test_results_data_info_copy():
    """Test the copy method of ResultsDataInfo."""
    info = ResultsDataInfo(technique=["echem"])
    info.parameters = {"analysis_type": "CV"}
    info_copy = info.copy()
    assert info_copy.technique == info.technique
    assert info_copy is not info


def test_results_data():
    ds = xr.Dataset({"peak_current": (["peak"], [0.1, 0.2])})
    res = ResultsData(data=ds)
    assert res.data.equals(ds)

    # Test summary and tables
    res.summary = {"max_i": 0.2}
    res.tables = {"peaks": ds}
    assert res.summary["max_i"] == 0.2
    assert "peaks" in res.tables


def test_metadata_mixin_get_precedence():
    info = RawDataInfo(sample_name="Sample")
    info.parameters = {"param1": "val1"}
    info.others = {"param1": "val2", "other1": "val3"}

    # Standard field takes precedence
    assert info.get("sample_name") == "Sample"
    # Parameters takes precedence over others
    assert info.get("param1") == "val1"
    # Others is checked last
    assert info.get("other1") == "val3"


def test_base_info_initialization_with_kwargs():
    """Test BaseInfo initialization with additional kwargs."""
    info = RawDataInfo(sample_name="Test", custom_field="custom_value")
    assert info.sample_name == "Test"
    assert info.get("custom_field") == "custom_value"

