import pytest
from pathlib import Path
from echemistpy.io.loaders import load, list_supported_formats, _register_loader
from echemistpy.io.structures import RawData, RawDataInfo
from echemistpy.io.plugin_manager import get_plugin_manager

# Define paths to example files
EXAMPLES_DIR = Path(__file__).parent.parent / "docs" / "examples"
BIOLOGIC_MPT = EXAMPLES_DIR / "echem" / "Biologic_EIS.mpt"
BIOLOGIC_GPCL = EXAMPLES_DIR / "echem" / "Biologic_GPCL.mpt"
LANHE_XLSX = EXAMPLES_DIR / "echem" / "LANHE_GPCL.xlsx"


def test_list_supported_formats():
    formats = list_supported_formats()
    assert ".mpt" in formats
    assert ".xlsx" in formats
    assert ".xye" in formats


@pytest.mark.skipif(not BIOLOGIC_MPT.exists(), reason="Example file not found")
def test_load_biologic_mpt():
    data, info = load(BIOLOGIC_MPT)
    assert isinstance(data, RawData)
    assert isinstance(info, RawDataInfo)
    # This file is EIS data, so it has different columns
    assert "Frequency/Hz" in data.data.data_vars or "Ewe/V" in data.data.data_vars
    assert info.instrument == "BioLogic"


@pytest.mark.skipif(not BIOLOGIC_GPCL.exists(), reason="Example file not found")
def test_load_biologic_gpcl():
    """Test loading BioLogic GPCL file."""
    data, info = load(BIOLOGIC_GPCL)
    assert isinstance(data, RawData)
    assert isinstance(info, RawDataInfo)
    assert info.instrument == "BioLogic"


@pytest.mark.skipif(not LANHE_XLSX.exists(), reason="Example file not found")
def test_load_lanhe_xlsx():
    # Lanhe might need openpyxl or similar
    try:
        data, info = load(LANHE_XLSX)
        assert isinstance(data, RawData)
        assert isinstance(info, RawDataInfo)
        assert "Ewe/V" in data.data.data_vars or "Voltage/V" in data.data.data_vars
    except ImportError:
        pytest.skip("Required library for XLSX not installed")


def test_load_non_existent():
    with pytest.raises(FileNotFoundError):
        load("non_existent_file.mpt")


def test_load_unsupported_extension():
    # Create a dummy file with unsupported extension
    dummy = Path("test.dummy")
    dummy.touch()
    try:
        with pytest.raises(ValueError, match="No loader registered"):
            load(dummy)
    finally:
        dummy.unlink()


def test_load_with_metadata_overrides():
    """Test load function with metadata override parameters."""
    if not BIOLOGIC_MPT.exists():
        pytest.skip("Example file not found")

    data, info = load(
        BIOLOGIC_MPT, sample_name="CustomSample", operator="TestUser", instrument="CustomInstrument", technique="custom_tech"
    )

    assert info.sample_name == "CustomSample"
    assert info.operator == "TestUser"
    # Instrument might be overridden by the reader, but technique should be set
    assert "custom_tech" in info.technique or info.technique == ["custom_tech"]


def test_load_without_standardization():
    """Test loading without automatic standardization."""
    if not BIOLOGIC_MPT.exists():
        pytest.skip("Example file not found")

    data, info = load(BIOLOGIC_MPT, standardize=False)
    assert isinstance(data, RawData)
    assert isinstance(info, RawDataInfo)


def test_load_with_format_override():
    """Test load with explicit format parameter."""
    if not BIOLOGIC_MPT.exists():
        pytest.skip("Example file not found")

    data, info = load(BIOLOGIC_MPT, fmt=".mpt")
    assert isinstance(data, RawData)


def test_register_loader():
    """Test _register_loader function."""

    class MockReader:
        def __init__(self, filepath, **kwargs):
            pass

        def load(self):
            import xarray as xr

            ds = xr.Dataset({"test": (["record"], [1.0])}, coords={"record": [0]})
            data = RawData(data=ds)
            info = RawDataInfo(sample_name="Mock", technique=["test"])
            return data, info

    # Register a new loader
    _register_loader([".mock"], MockReader)

    # Verify it's registered
    pm = get_plugin_manager()
    assert pm.get_loader(".mock") == MockReader


def test_list_supported_formats_descriptions():
    """Test that list_supported_formats returns proper descriptions."""
    formats = list_supported_formats()
    # Check that some formats have proper descriptions
    if ".mpt" in formats:
        assert "BioLogic" in formats[".mpt"]
    if ".xlsx" in formats:
        assert "LANHE" in formats[".xlsx"] or "Loaded by" in formats[".xlsx"]

