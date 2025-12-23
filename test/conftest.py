"""Shared pytest fixtures and configuration for echemistpy tests.

This module provides common test fixtures and utilities that can be used
across all test modules.
"""

import pytest
import xarray as xr
from pathlib import Path
from echemistpy.io.structures import RawData, RawDataInfo, ResultsData, ResultsDataInfo


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_dataset():
    """Create a simple xarray Dataset for testing."""
    return xr.Dataset(
        {"current": (["record"], [1.0, 2.0, 3.0]), "voltage": (["record"], [0.1, 0.2, 0.3])},
        coords={"record": [0, 1, 2]},
    )


@pytest.fixture
def sample_raw_data(sample_dataset):
    """Create a RawData instance for testing."""
    return RawData(data=sample_dataset)


@pytest.fixture
def sample_raw_data_info():
    """Create a RawDataInfo instance for testing."""
    return RawDataInfo(sample_name="TestSample", technique=["echem"], instrument="TestInstrument")


@pytest.fixture
def sample_results_data():
    """Create a ResultsData instance for testing."""
    ds = xr.Dataset({"peak_current": (["peak"], [0.1, 0.2, 0.3])}, coords={"peak": [0, 1, 2]})
    return ResultsData(data=ds)


@pytest.fixture
def sample_results_data_info():
    """Create a ResultsDataInfo instance for testing."""
    info = ResultsDataInfo(technique=["echem"])
    info.parameters = {"analysis_type": "CV", "scan_rate": 0.1}
    return info


# ============================================================================
# Example Data File Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def examples_dir():
    """Get the path to the examples directory."""
    return Path(__file__).parent.parent / "docs" / "examples"


@pytest.fixture(scope="session")
def biologic_eis_file(examples_dir):
    """Get the path to BioLogic EIS example file."""
    path = examples_dir / "echem" / "Biologic_EIS.mpt"
    if not path.exists():
        pytest.skip(f"Example file not found: {path}")
    return path


@pytest.fixture(scope="session")
def biologic_gpcl_file(examples_dir):
    """Get the path to BioLogic GPCL example file."""
    path = examples_dir / "echem" / "Biologic_GPCL.mpt"
    if not path.exists():
        pytest.skip(f"Example file not found: {path}")
    return path


@pytest.fixture(scope="session")
def lanhe_xlsx_file(examples_dir):
    """Get the path to LANHE XLSX example file."""
    path = examples_dir / "echem" / "LANHE_GPCL.xlsx"
    if not path.exists():
        pytest.skip(f"Example file not found: {path}")
    return path


# ============================================================================
# Test Utilities
# ============================================================================


@pytest.fixture
def assert_dataset_equal():
    """Fixture providing a function to assert two datasets are equal."""

    def _assert_equal(ds1: xr.Dataset, ds2: xr.Dataset, check_attrs: bool = True):
        """Assert two xarray Datasets are equal.

        Args:
            ds1: First dataset
            ds2: Second dataset
            check_attrs: Whether to check attributes
        """
        assert ds1.equals(ds2), f"Datasets are not equal:\nDS1: {ds1}\nDS2: {ds2}"
        if check_attrs:
            assert ds1.attrs == ds2.attrs, f"Attributes differ:\nDS1: {ds1.attrs}\nDS2: {ds2.attrs}"

    return _assert_equal


# ============================================================================
# Markers
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_data: marks tests that require example data files")
