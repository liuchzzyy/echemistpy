# echemistpy.io Module

Unified data loading and saving interface for scientific measurement data with a plugin-based architecture.

## Overview

The `io` module provides a simplified, extensible system for loading and saving scientific data. It uses a plugin architecture that makes it easy to add support for new file formats.

## Quick Start

### Loading Data

```python
from echemistpy.io import load

# Load a data file (format auto-detected)
raw_data, raw_info = load("experiment.mpt")
raw_data, raw_info = load("battery_test.xlsx")
raw_data, raw_info = load("data.csv")

# Force a specific format
raw_data, raw_info = load("data.txt", fmt="csv")
```

### Saving Data

```python
from echemistpy.io import save, save_measurement, save_results

# Save measurement data
save(measurement, measurement_info, "output.csv")
save(measurement, measurement_info, "output.h5", fmt="hdf5")

# Save analysis results
save_results(results, results_info, "analysis.csv")
```

### Standardizing Data

```python
from echemistpy.io import standardize_measurement

# Convert raw data to standardized measurement format
measurement, measurement_info = standardize_measurement(
    raw_data, 
    raw_info,
    technique_hint="cv"  # Optional: override auto-detection
)
```

## Architecture

### Core Components

1. **loader.py** - Unified loading interface
   - `load()`: Main function for loading files
   - `register_loader()`: Register custom file format loaders
   - `list_supported_formats()`: List all supported file formats

2. **saver.py** - Unified saving interface
   - `save()`: Main function for saving data
   - `save_measurement()`: Save measurement data
   - `save_results()`: Save analysis results

3. **standardizer.py** - Data standardization utilities
   - `standardize_measurement()`: Convert raw data to standardized format
   - `DataStandardizer`: Column name and unit standardization
   - `detect_technique()`: Auto-detect measurement technique

4. **structures.py** - Data container definitions
   - `RawData` + `RawDataInfo`: Unprocessed data from files
   - `Measurement` + `MeasurementInfo`: Standardized measurement data
   - `AnalysisResult` + `AnalysisResultInfo`: Processed analysis results

5. **plugin_manager.py** - Plugin registration and management
   - `IOPluginManager`: Manages loader and saver plugins
   - `get_plugin_manager()`: Get the global plugin manager

6. **plugin_specs.py** - Plugin interface specifications
   - `LoaderSpec`: Interface for file loaders
   - `SaverSpec`: Interface for file savers

### Data Flow

```
Raw File → load() → RawData + RawDataInfo
                         ↓
              standardize_measurement()
                         ↓
                Measurement + MeasurementInfo
                         ↓
                    Analysis
                         ↓
              AnalysisResult + AnalysisResultInfo
                         ↓
               save_results() → Output File
```

## Tested Plugins

The following readers have been tested and are ready for use:

### BiologicMPTReader.py
- **File formats**: `.mpt`, `.mpr`
- **Description**: BioLogic EC-Lab electrochemistry data files
- **Location**: `src/echemistpy/io/plugings/echem/BiologicMPTReader.py`
- **Features**: 
  - Metadata extraction from file headers
  - Support for multiple techniques (CV, GCD, PEIS, etc.)
  - Data cleaning and normalization

### LanheXLSXReader.py
- **File formats**: `.xlsx`, `.xls`
- **Description**: LANHE battery testing system data files
- **Location**: `src/echemistpy/io/plugings/echem/LanheXLSXReader.py`
- **Features**:
  - Multiple sheet parsing (Test information, Process info, Data)
  - Work mode table extraction
  - Cycle/step/record level data

## Adding New Plugins

To add support for a new file format, create a plugin class following this pattern:

### Step 1: Create Reader Class

Create a new file in `src/echemistpy/io/plugings/`:

```python
# my_reader.py
from pathlib import Path
from traitlets import HasTraits
import xarray as xr
import numpy as np

from echemistpy.io.plugin_specs import hookimpl
from echemistpy.io.structures import RawData, RawDataInfo


class MyDataReader(HasTraits):
    """Reader for MY_FORMAT files."""
    
    @staticmethod
    def load(filepath: Path) -> tuple[dict, dict]:
        """Load file and return metadata and data dictionaries."""
        # Your file reading logic here
        metadata = {"technique": "my_technique", "filename": filepath.stem}
        data_dict = {"column1": [...], "column2": [...]}
        return metadata, data_dict
```

### Step 2: Create Plugin Wrapper

```python
# my_plugin.py
from echemistpy.io.plugin_specs import hookimpl
from .my_reader import MyDataReader


class MyLoaderPlugin(HasTraits):
    """Plugin for MY_FORMAT files."""
    
    @hookimpl
    def get_supported_extensions(self) -> list[str]:
        return ["myformat", "mf"]
    
    @hookimpl
    def load_file(self, filepath: Path, **kwargs) -> tuple[RawData, RawDataInfo]:
        metadata, data_dict = MyDataReader.load(filepath)
        
        # Convert to xarray.Dataset
        dataset = self._dict_to_dataset(data_dict)
        
        return RawData(data=dataset), RawDataInfo(meta=metadata)
    
    @staticmethod
    def _dict_to_dataset(data_dict: dict) -> xr.Dataset:
        """Convert data dictionary to xarray.Dataset."""
        data_dict = {k: v for k, v in data_dict.items() if k != "_metadata"}
        n_rows = len(next(iter(data_dict.values())))
        
        data_vars = {}
        for name, values in data_dict.items():
            data_vars[name] = ("row", np.array(values))
        
        return xr.Dataset(
            data_vars=data_vars,
            coords={"row": np.arange(n_rows)}
        )
```

### Step 3: Register Plugin

Add to `loaders.py` in the `_initialize_default_plugins()` function:

```python
from echemistpy.io.plugings.my_plugin import MyLoaderPlugin
pm.register_plugin(MyLoaderPlugin(), name="my_format")
```

Or register dynamically:

```python
from echemistpy.io import register_loader
from my_plugin import MyLoaderPlugin

register_loader(MyLoaderPlugin(), name="my_format")
```

## API Reference

### Loading Functions

- **`load(path, fmt=None, **kwargs)`**: Load a data file
- **`load_data_file(path, fmt=None, **kwargs)`**: Alias for `load()` (backward compatibility)
- **`load_table(path, fmt=None, **kwargs)`**: Alias for `load()` (backward compatibility)
- **`register_loader(plugin, name=None)`**: Register a custom loader plugin

### Saving Functions

- **`save(measurement, info, path, fmt=None, **kwargs)`**: Save measurement data
- **`save_measurement(...)`**: Save Measurement object
- **`save_results(...)`**: Save AnalysisResult object

### Standardization Functions

- **`standardize_measurement(raw_data, raw_info, ...)`**: Convert raw data to standardized format
- **`detect_technique(dataset)`**: Auto-detect measurement technique from data

### Utility Functions

- **`get_file_info(path)`**: Get file information without loading
- **`list_supported_formats()`**: List all supported file formats
- **`get_plugin_manager()`**: Get the global plugin manager

## Supported File Formats

| Extension | Format | Description |
|-----------|--------|-------------|
| `.mpt`, `.mpr` | BioLogic | EC-Lab electrochemistry files |
| `.xlsx`, `.xls` | Excel | LANHE battery test files |
| `.csv`, `.txt`, `.tsv` | CSV/TSV | Delimited text files |
| `.h5`, `.hdf5`, `.hdf` | HDF5 | Hierarchical data format |
| `.nc`, `.nc4`, `.netcdf` | NetCDF | Network Common Data Form |
| `.json` | JSON | JavaScript Object Notation |

## Best Practices

1. **Always use `load()` for reading** - It automatically detects the file format
2. **Use standardization** - Convert raw data to Measurement format for consistency
3. **Preserve metadata** - Always save both data and info objects together
4. **Choose appropriate formats** - Use HDF5/NetCDF for large datasets, CSV for compatibility

## Examples

### Complete Workflow

```python
from echemistpy.io import load, standardize_measurement, save

# 1. Load raw data
raw_data, raw_info = load("experiment.mpt")

# 2. Standardize to measurement format
measurement, meas_info = standardize_measurement(raw_data, raw_info)

# 3. Save standardized data
save(measurement, meas_info, "standardized.h5")
```

### Working with Multiple Files

```python
from pathlib import Path
from echemistpy.io import load, standardize_measurement

# Load all .mpt files in a directory
data_dir = Path("experiments/")
measurements = []

for file in data_dir.glob("*.mpt"):
    raw_data, raw_info = load(file)
    measurement, meas_info = standardize_measurement(raw_data, raw_info)
    measurements.append((measurement, meas_info))
```

## Troubleshooting

### File Not Supported Error
```
ValueError: No loader found for extension 'xyz'
```
**Solution**: Check if the file format is supported with `list_supported_formats()` or register a custom loader.

### Missing Required Columns Warning
```
UserWarning: Missing required columns: ['Time/s', 'Current/mA']
```
**Solution**: The technique detection may be incorrect. Try explicitly setting `technique_hint` in `standardize_measurement()`.

### Plugin Already Registered Error
**Solution**: This is handled automatically in the latest version. Each plugin name can only be registered once.

## Migration Guide

### From Old API to New API

Old way:
```python
from echemistpy.io import load_data_file
raw_data, raw_info = load_data_file("data.mpt")
```

New way (recommended):
```python
from echemistpy.io import load
raw_data, raw_info = load("data.mpt")
```

Both work! The old function names are maintained for backward compatibility.
