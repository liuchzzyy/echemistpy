import pytest
import xarray as xr
import numpy as np
from echemistpy.io.standardizer import DataStandardizer, standardize_names
from echemistpy.io.structures import RawData, RawDataInfo


def test_data_standardizer_names():
    ds = xr.Dataset({"potential": (["record"], [1.0, 1.1]), "current": (["record"], [0.1, 0.2]), "time": (["record"], [0, 1])}, coords={"record": [0, 1]})

    standardizer = DataStandardizer(ds, techniques="echem")
    standardizer.standardize_column_names()
    std_ds = standardizer.get_dataset()

    assert "Ewe/V" in std_ds.data_vars
    assert "Current/mA" in std_ds.data_vars
    assert "Time/s" in std_ds.data_vars


def test_data_standardizer_units():
    ds = xr.Dataset({"current/uA": (["record"], [1000.0, 2000.0]), "time/min": (["record"], [1.0, 2.0]), "potential/mV": (["record"], [1000.0, 2000.0])}, coords={"record": [0, 1]})

    standardizer = DataStandardizer(ds, techniques="echem")
    standardizer.standardize_units()
    std_ds = standardizer.get_dataset()

    assert "current/mA" in std_ds.data_vars  # lowercase after unit conversion
    assert std_ds["current/mA"].values[0] == 1.0
    assert "time/s" in std_ds.data_vars
    assert std_ds["time/s"].values[0] == 60.0
    assert "potential/V" in std_ds.data_vars
    assert std_ds["potential/V"].values[0] == 1.0


def test_standardize_names_function():
    ds = xr.Dataset({"potential": (["record"], [1.0])}, coords={"record": [0]})
    rd = RawData(data=ds)
    info = RawDataInfo(technique="echem")

    std_rd, std_info = standardize_names(rd, info)
    assert "Ewe/V" in std_rd.data.data_vars
    assert std_info.technique == ["echem"]


def test_ensure_required_columns():
    ds = xr.Dataset({"Ewe/V": (["record"], [1.0])}, coords={"record": [0]})
    standardizer = DataStandardizer(ds, techniques="echem")
    standardizer.ensure_required_columns(["Current/mA"])
    std_ds = standardizer.get_dataset()

    assert "Current/mA" in std_ds.data_vars
    assert np.isnan(std_ds["Current/mA"].values[0])
