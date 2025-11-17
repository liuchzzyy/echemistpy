"""Comprehensive tests for the io module using real GPCL.mpr data."""

from pathlib import Path

import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
ECHEM_DIR = ROOT / "examples" / "echem"
GPCL_FILE = ECHEM_DIR / "Biologic_GPCL.mpr"


@pytest.fixture
def gpcl_measurement():
    """Load the GPCL.mpr file as a Measurement object."""
    from echemistpy.utils.external.echem import BiologicMPTReader

    reader = BiologicMPTReader()
    return reader.read(GPCL_FILE)


def test_biologic_reader_loads_gpcl_successfully(gpcl_measurement):
    """Test that the BiologicMPTReader can load GPCL.mpr file."""
    assert gpcl_measurement is not None
    assert gpcl_measurement.data is not None
    assert isinstance(gpcl_measurement.data, xr.Dataset)


def test_gpcl_has_expected_structure(gpcl_measurement):
    """Test that the loaded GPCL measurement has expected structure."""
    data = gpcl_measurement.data

    # Check dimensions
    assert "time_index" in data.dims
    assert data.sizes["time_index"] > 0

    # Check time coordinate exists
    assert "time/s" in data.variables

    # Check some expected electrochemistry variables
    expected_vars = ["Ewe/V", "Q charge/discharge/mA.h"]
    for var in expected_vars:
        assert var in data.variables, f"Expected variable {var} not found"


def test_gpcl_metadata_is_populated(gpcl_measurement):
    """Test that metadata is properly extracted from GPCL.mpr."""
    metadata = gpcl_measurement.metadata

    assert metadata.technique == "EC"
    assert metadata.sample_name is not None
    assert metadata.instrument == "BioLogic EC-Lab"

    # Check MPR-specific metadata
    extras = metadata.extras
    assert "mpr_version" in extras
    assert "mpr_columns" in extras
    assert isinstance(extras["mpr_columns"], tuple)


def test_gpcl_axes_are_defined(gpcl_measurement):
    """Test that axes are properly defined for GPCL measurement."""
    assert len(gpcl_measurement.axes) > 0

    time_axis = gpcl_measurement.axes[0]
    assert time_axis.name == "time/s"
    assert time_axis.unit == "s"
    assert time_axis.values is not None
    assert len(time_axis.values) == gpcl_measurement.data.sizes["time_index"]


def test_gpcl_data_values_are_numeric(gpcl_measurement):
    """Test that data values are numeric and not all zero."""
    data = gpcl_measurement.data

    # Check voltage data
    voltage = data["Ewe/V"].values
    assert voltage.dtype.kind == "f"  # float type
    assert not all(v == 0 for v in voltage)  # Not all zeros

    # Check charge data
    charge = data["Q charge/discharge/mA.h"].values
    assert charge.dtype.kind == "f"
    assert not all(c == 0 for c in charge)


def test_gpcl_time_is_monotonic_increasing(gpcl_measurement):
    """Test that time values are monotonically increasing."""
    import numpy as np

    time_values = gpcl_measurement.data["time/s"].values
    time_diffs = np.diff(time_values)
    assert np.all(time_diffs >= 0), "Time values should be monotonically increasing"


def test_measurement_copy_works(gpcl_measurement):
    """Test that measurement.copy() creates an independent copy."""
    copied = gpcl_measurement.copy()

    assert copied is not gpcl_measurement
    assert copied.data is not gpcl_measurement.data
    assert copied.metadata is not gpcl_measurement.metadata

    # Verify data independence
    original_shape = gpcl_measurement.data.sizes["time_index"]
    copied_shape = copied.data.sizes["time_index"]
    assert original_shape == copied_shape


def test_measurement_require_variables(gpcl_measurement):
    """Test the require_variables method."""
    # Should not raise for existing variables
    gpcl_measurement.require_variables(["time/s", "Ewe/V"])

    # Should raise for non-existent variables
    with pytest.raises(ValueError, match="missing required variables"):
        gpcl_measurement.require_variables(["nonexistent_column"])


def test_io_module_exports(gpcl_measurement):
    """Test that key io module components are accessible."""
    from echemistpy.io import Axis, Measurement, MeasurementMetadata

    # Verify types
    assert isinstance(gpcl_measurement, Measurement)
    assert isinstance(gpcl_measurement.metadata, MeasurementMetadata)
    assert all(isinstance(ax, Axis) for ax in gpcl_measurement.axes)
