import pytest
from pathlib import Path
from echemistpy.io.loaders import load, list_supported_formats
from echemistpy.io.structures import RawData, RawDataInfo

# Define paths to example files
EXAMPLES_DIR = Path(__file__).parent.parent / "docs" / "examples"
BIOLOGIC_MPT = EXAMPLES_DIR / "echem" / "Biologic_EIS.mpt"
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
