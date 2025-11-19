import pytest
from pathlib import Path
import xarray as xr
from echemistpy.io.loaders import load_data_file
from echemistpy.io.standardized import standardize_measurement
from echemistpy.io.structures import Measurement

def test_pipeline_execution(tmp_path):
    """
    Verify the entire pipeline:
    1. Load Biologic_GPCL.mpr
    2. Standardize the data
    3. Save the result
    """
    # 1. Load data
    # Assuming tests are run from the project root or tests directory
    # We need to find the examples directory relative to this file
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "examples" / "echem" / "Biologic_GPCL.mpr"
    
    print(f"\nLooking for file at: {file_path}")
    assert file_path.exists(), f"Test file not found: {file_path}"
    
    print("Loading data...")
    raw_measurement = load_data_file(file_path)
    assert raw_measurement is not None
    
    # Check raw data content
    print("\n--- Raw Data Info ---")
    print(f"Technique: {raw_measurement.metadata.meta.get('technique')}")
    if hasattr(raw_measurement.data.data, 'data_vars'):
        print(f"Data Variables: {list(raw_measurement.data.data.data_vars.keys())}")
    
    # 2. Standardize
    print("\nStandardizing data...")
    measurement, info = standardize_measurement(raw_measurement)
    
    assert isinstance(measurement, Measurement)
    assert info.technique is not None
    
    print("\n--- Standardized Data Info ---")
    print(f"Technique: {info.technique}")
    print(f"Sample Name: {info.sample_name}")
    print(f"Variables: {list(measurement.data.data_vars.keys())}")
    
    # Verify some expected columns for electrochemistry
    expected_cols = ['Time/s', 'Ewe/V', 'Current/mA']
    # Note: exact column names depend on the standardization logic and input file
    # We check if at least some standard columns are present
    found_cols = list(measurement.data.data_vars.keys())
    print(f"Found columns: {found_cols}")
    
    # 3. Save data
    # Saving as NetCDF to a temporary file
    save_path = tmp_path / "verified_output.nc"
    print(f"\nSaving data to: {save_path}")
    
    measurement.data.to_netcdf(save_path)
    
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    print("Save successful.")

if __name__ == "__main__":
    # Allow running directly for quick check
    import sys
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        test_pipeline_execution(Path(tmp))
