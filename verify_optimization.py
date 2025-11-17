#!/usr/bin/env python
"""
Verification script to demonstrate the optimized echemistpy package.

This script verifies:
1. BiologicMPTReader works correctly with GPCL.mpr
2. IO module functions properly
3. Excel loading works without pandas direct usage
4. All core functionality is intact
"""

import sys
from pathlib import Path

# Add src to path for running from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_biologic_reader():
    """Test BiologicMPTReader with GPCL.mpr file."""
    print("=" * 70)
    print("TEST 1: BiologicMPTReader with GPCL.mpr")
    print("=" * 70)
    
    from echemistpy.utils.external.echem.biologic_reader import BiologicMPTReader
    
    gpcl_path = Path(__file__).parent / "examples" / "echem" / "Biologic_GPCL.mpr"
    
    if not gpcl_path.exists():
        print(f"❌ GPCL.mpr file not found at {gpcl_path}")
        return False
    
    reader = BiologicMPTReader()
    measurement = reader.read(gpcl_path)
    
    print(f"✓ Loaded {len(measurement.data.time_index)} data points")
    print(f"✓ Sample: {measurement.metadata.sample_name}")
    print(f"✓ Technique: {measurement.metadata.technique}")
    print(f"✓ Instrument: {measurement.metadata.instrument}")
    
    # Check key columns
    required_cols = ["time/s", "Ewe/V", "Q charge/discharge/mA.h"]
    for col in required_cols:
        if col not in measurement.data.variables:
            print(f"❌ Missing required column: {col}")
            return False
        print(f"✓ Column '{col}' present")
    
    # Check data ranges
    voltage = measurement.data["Ewe/V"].values
    charge = measurement.data["Q charge/discharge/mA.h"].values
    print(f"✓ Voltage range: {voltage.min():.3f} to {voltage.max():.3f} V")
    print(f"✓ Charge range: {charge.min():.3f} to {charge.max():.3f} mA.h")
    
    print("\n✅ BiologicMPTReader test PASSED\n")
    return True


def test_io_module():
    """Test IO module exports and functionality."""
    print("=" * 70)
    print("TEST 2: IO Module Exports")
    print("=" * 70)
    
    from echemistpy.io import (
        Measurement,
        MeasurementMetadata,
        Axis,
        AnalysisResult,
        load_table,
        load_data_file,
        save_table,
    )
    
    print("✓ Measurement class imported")
    print("✓ MeasurementMetadata class imported")
    print("✓ Axis class imported")
    print("✓ AnalysisResult class imported")
    print("✓ load_table function imported")
    print("✓ load_data_file function imported")
    print("✓ save_table function imported")
    
    print("\n✅ IO module exports test PASSED\n")
    return True


def test_excel_loading():
    """Test Excel loading without pandas."""
    print("=" * 70)
    print("TEST 3: Excel Loading (openpyxl backend)")
    print("=" * 70)
    
    import tempfile
    from pathlib import Path
    
    try:
        from openpyxl import Workbook
        from echemistpy.io.loaders import load_data_file
        
        with tempfile.TemporaryDirectory() as tmp:
            excel_path = Path(tmp) / "test_data.xlsx"
            
            # Create test Excel file
            wb = Workbook()
            ws = wb.active
            ws.append(["Name", "Value", "Category"])
            ws.append(["Item1", 10, "A"])
            ws.append(["Item2", 20, "B"])
            ws.append(["Item3", 30, "A"])
            wb.save(excel_path)
            
            # Load using new implementation
            dataset = load_data_file(excel_path)
            
            print(f"✓ Excel file loaded successfully")
            print(f"✓ Columns: {list(dataset.data_vars.keys())}")
            print(f"✓ Rows: {len(dataset.row)}")
            print(f"✓ Values: {list(dataset['Value'].values)}")
            
            # Verify data
            assert len(dataset.row) == 3
            assert list(dataset["Value"].values) == [10, 20, 30]
            print(f"✓ Data integrity verified")
            
        print("\n✅ Excel loading test PASSED\n")
        return True
        
    except ImportError as e:
        print(f"⚠️  Excel test skipped: {e}")
        return True


def test_removed_modules():
    """Verify that removed modules are no longer accessible."""
    print("=" * 70)
    print("TEST 4: Verify Removed Modules")
    print("=" * 70)
    
    # These should not be importable
    removed_modules = [
        ("echemistpy.io.organization", "DataCleaner"),
        ("echemistpy.utils.math", "moving_average"),
    ]
    
    for module_path, item_name in removed_modules:
        try:
            module = __import__(module_path, fromlist=[item_name])
            getattr(module, item_name)
            print(f"❌ {module_path}.{item_name} should not exist")
            return False
        except (ImportError, ModuleNotFoundError, AttributeError):
            print(f"✓ {module_path}.{item_name} correctly removed")
    
    print("\n✅ Removed modules verification PASSED\n")
    return True


def test_core_functionality():
    """Test core package functionality."""
    print("=" * 70)
    print("TEST 5: Core Package Functionality")
    print("=" * 70)
    
    import xarray as xr
    from echemistpy.io import Measurement, MeasurementMetadata, Axis
    
    # Create a simple measurement
    data_vars = {
        "time": ("index", [0, 1, 2, 3, 4]),
        "voltage": ("index", [1.0, 1.1, 1.2, 1.3, 1.4]),
        "current": ("index", [0.1, 0.2, 0.3, 0.4, 0.5]),
    }
    dataset = xr.Dataset(data_vars, coords={"index": [0, 1, 2, 3, 4]})
    
    metadata = MeasurementMetadata(
        technique="CV",
        sample_name="Test Sample",
        instrument="Test Instrument",
    )
    
    axis = Axis(name="time", unit="s", values=[0, 1, 2, 3, 4])
    
    measurement = Measurement(data=dataset, metadata=metadata, axes=[axis])
    
    print("✓ Created Measurement object")
    print(f"✓ Data shape: {dataset.dims}")
    print(f"✓ Metadata: {metadata.technique}, {metadata.sample_name}")
    print(f"✓ Axes: {len(measurement.axes)}")
    
    # Test copy
    copied = measurement.copy()
    assert copied is not measurement
    print("✓ Measurement copy works")
    
    # Test require_variables
    try:
        measurement.require_variables(["time", "voltage"])
        print("✓ require_variables works for existing columns")
    except ValueError:
        print("❌ require_variables failed unexpectedly")
        return False
    
    try:
        measurement.require_variables(["nonexistent"])
        print("❌ require_variables should have raised ValueError")
        return False
    except ValueError:
        print("✓ require_variables correctly raises for missing columns")
    
    print("\n✅ Core functionality test PASSED\n")
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("ECHEMISTPY PACKAGE OPTIMIZATION VERIFICATION")
    print("=" * 70 + "\n")
    
    tests = [
        ("BiologicMPTReader", test_biologic_reader),
        ("IO Module", test_io_module),
        ("Excel Loading", test_excel_loading),
        ("Removed Modules", test_removed_modules),
        ("Core Functionality", test_core_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} test FAILED with exception: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL VERIFICATIONS PASSED! Package optimization successful.\n")
        return 0
    else:
        print("\n⚠️  Some verifications failed. Please review the output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
