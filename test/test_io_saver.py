import pytest
import xarray as xr
import pandas as pd
import json
from pathlib import Path
from echemistpy.io.saver import save_info, save_data, save_combined, _sanitize_dataset
from echemistpy.io.structures import RawData, RawDataInfo


def test_sanitize_dataset():
    ds = xr.Dataset({"current/mA": (["record"], [1.0])})
    sanitized = _sanitize_dataset(ds)
    assert "current_mA" in sanitized.data_vars
    assert "current/mA" not in sanitized.data_vars


def test_save_info(tmp_path):
    info = RawDataInfo(sample_name="Test", technique="echem")
    path = tmp_path / "info.dat"
    save_info(info, path)

    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["sample_name"] == "Test"
    assert data["technique"] == ["echem"]  # technique is a list


def test_save_data_csv(tmp_path):
    ds = xr.Dataset({"current": (["record"], [1.0, 2.0])}, coords={"record": [0, 1]})
    rd = RawData(data=ds)
    path = tmp_path / "data.csv"
    save_data(rd, path)

    assert path.exists()
    df = pd.read_csv(path)
    assert "current" in df.columns
    assert len(df) == 2


def test_save_data_nc(tmp_path):
    ds = xr.Dataset({"current": (["record"], [1.0, 2.0])}, coords={"record": [0, 1]})
    rd = RawData(data=ds)
    path = tmp_path / "data.nc"
    # NetCDF might need h5netcdf or netCDF4
    try:
        save_data(rd, path)
        assert path.exists()
        ds_loaded = xr.open_dataset(path)
        assert "current" in ds_loaded.data_vars
    except (ImportError, ModuleNotFoundError):
        pytest.skip("h5netcdf or netCDF4 not installed")


def test_save_combined(tmp_path):
    ds = xr.Dataset({"current": (["record"], [1.0, 2.0])}, coords={"record": [0, 1]})
    rd = RawData(data=ds)
    info = RawDataInfo(sample_name="Test", technique="echem")
    path = tmp_path / "combined.nc"

    try:
        save_combined(rd, info, path)
        assert path.exists()
        ds_loaded = xr.open_dataset(path)
        assert ds_loaded.attrs["sample_name"] == "Test"
        assert "current" in ds_loaded.data_vars
    except (ImportError, ModuleNotFoundError):
        pytest.skip("h5netcdf or netCDF4 not installed")
