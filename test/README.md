# Test Suite for echemistpy

This directory contains comprehensive tests for the echemistpy library, focusing on the `io` module.

## Test Structure

### Test Files

- `test_io_structures.py` - Tests for data structures (RawData, RawDataInfo, ResultsData, ResultsDataInfo)
- `test_io_loaders.py` - Tests for data loading functionality
- `test_io_saver.py` - Tests for data saving functionality
- `test_io_standardizer.py` - Tests for data standardization
- `test_io_plugin_manager.py` - Tests for plugin management system
- `conftest.py` - Shared fixtures and test configuration

## Running Tests

### Run all tests
```bash
python -m pytest test/
```

### Run tests with verbose output
```bash
python -m pytest test/ -v
```

### Run tests with coverage report
```bash
python -m pytest test/ --cov=src/echemistpy/io --cov-report=term-missing
```

### Generate HTML coverage report
```bash
python -m pytest test/ --cov=src/echemistpy/io --cov-report=html
```

### Run specific test file
```bash
python -m pytest test/test_io_structures.py -v
```

### Run tests matching a pattern
```bash
python -m pytest test/ -k "standardizer" -v
```

### Run tests excluding slow tests
```bash
python -m pytest test/ -m "not slow"
```

## Test Coverage

Current coverage statistics (as of last update):

- **Total Coverage**: 78%
- **structures.py**: 96%
- **saver.py**: 96%
- **plugin_manager.py**: 97%
- **standardizer.py**: 92%
- **loaders.py**: 88%

## Writing New Tests

### Using Fixtures

The `conftest.py` file provides several useful fixtures:

```python
def test_example(sample_dataset, sample_raw_data_info):
    # Use the fixtures directly
    data = RawData(data=sample_dataset)
    assert data.data.equals(sample_dataset)
```

### Test Organization

Each test should:
1. Have a clear, descriptive name starting with `test_`
2. Test a single function or feature
3. Use appropriate fixtures for test data
4. Include docstrings for complex tests
5. Use markers for categorization (e.g., `@pytest.mark.slow`)

### Example Test

```python
def test_load_data_with_metadata(biologic_eis_file):
    """Test loading data with custom metadata overrides."""
    data, info = load(
        biologic_eis_file,
        sample_name="CustomSample",
        operator="TestUser"
    )
    
    assert info.sample_name == "CustomSample"
    assert info.operator == "TestUser"
```

## Test Data

Test data files are located in `docs/examples/`:
- `echem/Biologic_EIS.mpt` - BioLogic EIS data
- `echem/Biologic_GPCL.mpt` - BioLogic galvanostatic cycling data
- `echem/LANHE_GPCL.xlsx` - LANHE battery test data
- `opeando_xrd/` - XRD data files

## CI/CD Integration

These tests are designed to run in CI/CD pipelines. To integrate:

1. Ensure all dependencies are installed: `pip install -e ".[test]"`
2. Run tests with coverage: `pytest test/ --cov=src/echemistpy/io`
3. Generate coverage reports for CI tools

## Troubleshooting

### Missing Dependencies

If tests fail due to missing dependencies:
```bash
pip install -e ".[test]"
```

### Example Files Not Found

Some tests require example data files. These tests will be skipped if files are missing.
To see which tests are skipped:
```bash
python -m pytest test/ -v -rs
```

### Slow Tests

Some tests may be slow due to file I/O operations. Mark slow tests:
```python
@pytest.mark.slow
def test_large_file_processing():
    # Test code here
    pass
```

Then run without slow tests:
```bash
python -m pytest test/ -m "not slow"
```
