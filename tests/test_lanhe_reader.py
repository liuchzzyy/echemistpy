import pytest
from pathlib import Path
from echemistpy.utils.external.echem.lanhe_reader import LanheReader
from echemistpy.io.structures import RawData, RawDataInfo

# Define paths to example files
EXAMPLES_DIR = Path(__file__).parents[1] / "examples" / "echem"
XLSX_FILE = EXAMPLES_DIR / "LANHE_GPCL.xlsx"

def test_read_xlsx():
    """Test reading a Lanhe .xlsx file."""
    assert XLSX_FILE.exists(), f"Test file not found: {XLSX_FILE}"
    
    reader = LanheReader(XLSX_FILE)
    raw_data, raw_data_info = reader.read()
    
    assert isinstance(raw_data, RawData)
    assert isinstance(raw_data_info, RawDataInfo)
    
    # Check data content
    assert raw_data.data is not None
    assert "row" in raw_data.data.coords
    
    # Check metadata
    assert raw_data_info.meta is not None
    assert "filename" in raw_data_info.meta
    assert raw_data_info.meta["filename"] == XLSX_FILE.name
    
    # Check for standardized columns (if the file has them)
    # We check for at least one common column to verify standardization worked
    # Note: We don't know exact content of the example file, but we expect some standard echem columns
    vars = list(raw_data.data.data_vars.keys())
    print(f"Found variables: {vars}")
    
    # Check if we have time or voltage or current
    possible_standard_cols = ["time/s", "Ewe/V", "I/mA", "Capacity/mAh"]
    assert any(col in vars for col in possible_standard_cols), f"No standard columns found. Vars: {vars}"
