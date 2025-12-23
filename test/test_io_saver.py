import pytest
import xarray as xr
import pandas as pd
import json
from pathlib import Path
from echemistpy.io.saver import save_info, save_data, save_combined, _sanitize_dataset, _to_list
from echemistpy.io.structures import RawData, RawDataInfo, ResultsData, ResultsDataInfo


def test_sanitize_dataset():
    ds = xr.Dataset({"current/mA": (["record"], [1.0])})
    sanitized = _sanitize_dataset(ds)
    assert "current_mA" in sanitized.data_vars
    assert "current/mA" not in sanitized.data_vars


def test_sanitize_dataset_coords():
    """Test that _sanitize_dataset also sanitizes coordinate names."""
    ds = xr.Dataset({"current": (["record"], [1.0])}, coords={"time/s": (["record"], [0])})
    sanitized = _sanitize_dataset(ds)
    assert "time_s" in sanitized.coords
    assert "time/s" not in sanitized.coords


def test_to_list():
    """Test the _to_list utility function."""
    # Test with single object
    assert _to_list(1) == [1]
    # Test with list
    assert _to_list([1, 2]) == [1, 2]
    # Test with tuple
    assert _to_list((1, 2)) == [1, 2]


def test_save_info(tmp_path):
    info = RawDataInfo(sample_name="Test", technique="echem")
    path = tmp_path / "info.dat"
    save_info(info, path)

    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["sample_name"] == "Test"
    assert data["technique"] == ["echem"]  # technique is a list


def test_save_info_multiple(tmp_path):
    """Test saving multiple Info objects."""
    info1 = RawDataInfo(sample_name="Test1", technique=["echem"])
    info2 = RawDataInfo(sample_name="Test2", technique=["xrd"])
    path = tmp_path / "infos.dat"
    save_info([info1, info2], path)

    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["sample_name"] == "Test1"
    assert data[1]["sample_name"] == "Test2"


def test_save_data_csv(tmp_path):
    ds = xr.Dataset({"current": (["record"], [1.0, 2.0])}, coords={"record": [0, 1]})
    rd = RawData(data=ds)
    path = tmp_path / "data.csv"
    save_data(rd, path)

    assert path.exists()
    df = pd.read_csv(path)
    assert "current" in df.columns
    assert len(df) == 2


def test_save_data_csv_with_fmt_override(tmp_path):
    """Test save_data with format override parameter."""
    ds = xr.Dataset({"current": (["record"], [1.0, 2.0])}, coords={"record": [0, 1]})
    rd = RawData(data=ds)
    path = tmp_path / "data.txt"
    save_data(rd, path, fmt="csv")

    assert path.exists()
    df = pd.read_csv(path)
    assert "current" in df.columns


def test_save_data_multiple_csv(tmp_path):
    """Test saving multiple Data objects to CSV."""
    ds1 = xr.Dataset({"current": (["record"], [1.0])}, coords={"record": [0]})
    ds2 = xr.Dataset({"current": (["record"], [2.0])}, coords={"record": [1]})
    rd1 = RawData(data=ds1)
    rd2 = RawData(data=ds2)
    path = tmp_path / "data.csv"
    save_data([rd1, rd2], path)

    assert path.exists()
    df = pd.read_csv(path)
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


def test_save_data_unsupported_format(tmp_path):
    """Test that save_data raises error for unsupported format."""
    ds = xr.Dataset({"current": (["record"], [1.0])}, coords={"record": [0]})
    rd = RawData(data=ds)
    path = tmp_path / "data.txt"

    with pytest.raises(ValueError, match="Unsupported format"):
        save_data(rd, path, fmt="txt")


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


def test_save_combined_multiple(tmp_path):
    """Test saving multiple data and info objects."""
    ds1 = xr.Dataset({"current": (["record"], [1.0])}, coords={"record": [0]})
    ds2 = xr.Dataset({"voltage": (["record"], [2.0])}, coords={"record": [0]})
    rd1 = RawData(data=ds1)
    rd2 = RawData(data=ds2)
    info1 = RawDataInfo(sample_name="Test1", technique=["echem"])
    info2 = RawDataInfo(sample_name="Test2", technique=["echem"])
    path = tmp_path / "combined.nc"

    try:
        save_combined([rd1, rd2], [info1, info2], path)
        assert path.exists()
        ds_loaded = xr.open_dataset(path)
        # Should merge both datasets
        assert "current" in ds_loaded.data_vars or "voltage" in ds_loaded.data_vars
    except (ImportError, ModuleNotFoundError):
        pytest.skip("h5netcdf or netCDF4 not installed")


def test_save_combined_single_info_multiple_data(tmp_path):
    """Test saving multiple data objects with a single info."""
    ds1 = xr.Dataset({"current": (["record"], [1.0])}, coords={"record": [0]})
    ds2 = xr.Dataset({"voltage": (["record"], [2.0])}, coords={"record": [0]})
    rd1 = RawData(data=ds1)
    rd2 = RawData(data=ds2)
    info = RawDataInfo(sample_name="Test", technique=["echem"])
    path = tmp_path / "combined.nc"

    try:
        save_combined([rd1, rd2], info, path)
        assert path.exists()
    except (ImportError, ModuleNotFoundError):
        pytest.skip("h5netcdf or netCDF4 not installed")


def test_save_combined_mismatched_lengths(tmp_path):
    """Test that save_combined raises error when data and info lengths don't match."""
    ds1 = xr.Dataset({"current": (["record"], [1.0])}, coords={"record": [0]})
    ds2 = xr.Dataset({"voltage": (["record"], [2.0])}, coords={"record": [0]})
    rd1 = RawData(data=ds1)
    rd2 = RawData(data=ds2)
    info = RawDataInfo(sample_name="Test", technique=["echem"])
    path = tmp_path / "combined.nc"

    # 2 data objects but only 1 info should work (uses single info for all)
    # But 2 data and 0 info should fail
    with pytest.raises(ValueError, match="Number of data objects and info objects must match"):
        save_combined([rd1, rd2], [], path)

