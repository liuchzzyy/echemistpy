import pytest
import xarray as xr
import pandas as pd
import numpy as np
from echemistpy.io.structures import RawData, RawDataInfo, ResultsData, ResultsDataInfo


def test_raw_data_info():
    info = RawDataInfo(sample_name="TestSample", technique="echem", instrument="BioLogic")
    assert info.sample_name == "TestSample"
    assert info.technique == "echem"
    assert info.instrument == "BioLogic"

    # Test to_dict
    d = info.to_dict()
    assert d["sample_name"] == "TestSample"
    assert d["technique"] == "echem"

    # Test get
    assert info.get("sample_name") == "TestSample"
    assert info.get("non_existent", "default") == "default"

    # Test update
    info.update({"operator": "Alice", "custom_param": 123})
    assert info.operator == "Alice"
    assert info.parameters["custom_param"] == 123
    assert info.get("custom_param") == 123


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


def test_results_data_info():
    info = ResultsDataInfo(technique="echem", analysis_type="CV")
    assert info.technique == "echem"
    assert info.analysis_type == "CV"

    d = info.to_dict()
    assert d["analysis_type"] == "CV"


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
