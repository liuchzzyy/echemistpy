import pytest
from pathlib import Path
from echemistpy.utils.external.echem.biologic_reader import BiologicMPTReader
from echemistpy.io.structures import RawData, RawDataInfo

# Define paths to example files
EXAMPLES_DIR = Path(__file__).parents[1] / "examples" / "echem"
MPT_FILE = EXAMPLES_DIR / "Biologic_GPCL.mpt"
MPR_FILE = EXAMPLES_DIR / "Biologic_GPCL.mpr"

def test_read_mpt():
    """Test reading a .mpt file."""
    assert MPT_FILE.exists(), f"Test file not found: {MPT_FILE}"
    
    reader = BiologicMPTReader(MPT_FILE)
    raw_data, raw_data_info = reader.read()
    
    assert isinstance(raw_data, RawData)
    assert isinstance(raw_data_info, RawDataInfo)
    
    # Check data content
    assert raw_data.data is not None
    assert "row" in raw_data.data.coords
    
    # Check metadata
    assert raw_data_info.meta is not None
    assert "header" in raw_data_info.meta

def test_read_mpr():
    """Test reading a .mpr file."""
    assert MPR_FILE.exists(), f"Test file not found: {MPR_FILE}"
    
    reader = BiologicMPTReader(MPR_FILE)
    raw_data, raw_data_info = reader.read()
    
    assert isinstance(raw_data, RawData)
    assert isinstance(raw_data_info, RawDataInfo)
    
    # Check data content
    assert raw_data.data is not None
    assert "row" in raw_data.data.coords
    
    # Check metadata
    assert raw_data_info.meta is not None
    assert "start_date" in raw_data_info.meta
    assert "version" in raw_data_info.meta
