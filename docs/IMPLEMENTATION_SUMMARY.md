# Implementation Summary: Pluggy + Traitlets IO System

## Problem Statement

Update the echemistpy io module to:
1. Use existing plugins from `src/echemistpy/io/plugings/echem` (BiologicMPTReader.py, LanheXLSXReader.py)
2. Implement pluggy for managing different plugins
3. Use traitlets for parameter control
4. Create unified loader.py and saver.py interfaces

## Solution Architecture

### 1. Plugin Specification Layer (`plugin_specs.py`)

Created hook specifications using pluggy:
- `LoaderSpec`: Defines hooks for file loaders (`get_supported_extensions`, `can_load`, `load_file`)
- `SaverSpec`: Defines hooks for file savers (`get_supported_formats`, `save_data`)

### 2. Plugin Manager (`plugin_manager.py`)

Implemented `IOPluginManager` class:
- Manages plugin registration and discovery
- Maintains extension → loader and format → saver mappings
- Provides `load_file()` and `save_data()` dispatch methods
- Global singleton instance via `get_plugin_manager()`

### 3. Plugin Wrappers

#### Echem-Specific Plugins
- **`biologic_plugin.py`**: Wraps `BiologicDataReader` from BiologicMPTReader.py
  - Supports .mpt and .mpr files
  - Uses traitlets for configuration (encoding parameter)
  - Converts BiologicDataReader output to RawData/RawDataInfo format
  
- **`lanhe_plugin.py`**: Wraps `XLSXDataReader` from LanheXLSXReader.py
  - Supports .xlsx and .xls files for LANHE battery test data
  - Uses traitlets for configuration
  - Converts XLSXDataReader output to RawData/RawDataInfo format

#### Generic Format Plugins (`generic_loaders.py`, `generic_savers.py`)
- **CSVLoaderPlugin**: CSV/TSV file loading with configurable delimiter
- **ExcelLoaderPlugin**: Generic Excel file loading (non-LANHE files)
- **HDF5LoaderPlugin**: HDF5/NetCDF file loading
- **CSVSaverPlugin**: CSV/TSV file saving
- **HDF5SaverPlugin**: HDF5/NetCDF file saving with metadata serialization
- **JSONSaverPlugin**: JSON file saving

All plugins use traitlets HasTraits base class for parameter validation.

### 4. Unified Interface Layer

#### `loaders.py`
- `_initialize_default_plugins()`: Registers all built-in plugins on module import
- `_load()`: Main loading function that delegates to plugin manager
- `register_loader()`: Public API for registering custom plugins
- `standardize_measurement()`: Converts RawData to standardized Measurement format
- Backward compatible public aliases: `load_data_file = _load`, `load_table = _load`

#### `saver.py`
- `save_measurement()`: Saves Measurement objects using plugin manager
- `save_results()`: Saves AnalysisResult objects with optional measurement data
- Special handling for HDF5/NetCDF to support multiple groups

### 5. Data Structures (`structures.py`)

No changes needed - existing structures work perfectly:
- `RawData` + `RawDataInfo`: Container for raw file data
- `Measurement` + `MeasurementInfo`: Standardized measurement data
- `AnalysisResult` + `AnalysisResultInfo`: Analysis results

All use `xarray.Dataset` as the underlying data container.

## Key Design Decisions

### 1. Plugin Registration Strategy
Plugins are automatically registered on module import via `_initialize_default_plugins()`:
```python
# In loaders.py
def _initialize_default_plugins():
    pm = get_plugin_manager()
    # Register echem-specific plugins
    pm.register_plugin(BiologicLoaderPlugin(), name="biologic")
    pm.register_plugin(LanheLoaderPlugin(), name="lanhe")
    # Register generic plugins
    pm.register_plugin(CSVLoaderPlugin(), name="csv")
    ...

_initialize_default_plugins()  # Called on import
```

### 2. Wrapper Pattern for Existing Readers
Rather than modifying BiologicMPTReader.py and LanheXLSXReader.py directly, we created thin wrapper plugins:
```python
class BiologicLoaderPlugin(HasTraits):
    @hookimpl
    def load_file(self, filepath, **kwargs):
        # Use existing BiologicDataReader
        metadata, data_dict = BiologicDataReader.load(filepath)
        # Convert to RawData format
        dataset = self._dict_to_dataset(data_dict)
        return RawData(data=dataset), RawDataInfo(meta=metadata)
```

This approach:
- Preserves existing code
- Adds pluggy interface layer
- Minimal code changes required

### 3. Traitlets for Configuration
All plugins inherit from `HasTraits` and define configuration parameters:
```python
class CSVLoaderPlugin(HasTraits):
    delimiter = Unicode(default_value=",", help="Delimiter character").tag(config=True)
    encoding = Unicode(default_value="utf-8", help="File encoding").tag(config=True)
```

Benefits:
- Type validation
- Default values
- Documentation via help text
- Runtime configuration via kwargs

### 4. Metadata Handling in Savers
For HDF5/NetCDF, complex nested dictionaries can't be saved as attributes. Solution:
```python
# Serialize entire metadata as JSON
dataset.attrs["echemistpy_metadata"] = json.dumps(metadata)

# Also save simple top-level fields directly
for k, v in metadata.items():
    if isinstance(v, (str, int, float, bool)):
        dataset.attrs[k] = v
```

## API Examples

### Loading Data
```python
from echemistpy.io import load_data_file

# Automatic format detection
raw_data, raw_data_info = load_data_file("data.mpt")

# Explicit format
raw_data, raw_data_info = load_data_file("data.txt", fmt="csv")
```

### Saving Data
```python
from echemistpy.io import save_measurement

# Save to CSV
save_measurement(measurement, measurement_info, "output.csv")

# Save to NetCDF
save_measurement(measurement, measurement_info, "output.nc")
```

### Custom Plugin
```python
from echemistpy.io import register_loader
from echemistpy.io.plugin_specs import hookimpl
from traitlets import HasTraits

class MyLoaderPlugin(HasTraits):
    @hookimpl
    def get_supported_extensions(self):
        return ["xyz"]
    
    @hookimpl
    def load_file(self, filepath, **kwargs):
        # Custom loading logic
        ...
        return raw_data, raw_data_info

register_loader(MyLoaderPlugin(), name="my_loader")
```

## Testing

Comprehensive test suite validates:
1. Plugin manager initialization (13 loaders, 10 savers)
2. Format listing and discovery
3. CSV round-trip (load → save → load)
4. NetCDF round-trip with metadata preservation
5. JSON saving
6. Traitlets configuration

All tests pass successfully.

## Migration Notes

### Breaking Changes
None - API is backward compatible

### New Features
- `get_plugin_manager()`: Access plugin manager for advanced usage
- `register_loader()`: Register custom plugins
- Plugin-based architecture allows easy extensibility

### Files Modified
- `src/echemistpy/io/loaders.py`: Refactored to use plugin manager
- `src/echemistpy/io/saver.py`: Refactored to use plugin manager
- `src/echemistpy/io/__init__.py`: Export plugin management functions
- `src/echemistpy/__init__.py`: Fix import from non-existent core module

### Files Created
- `src/echemistpy/io/plugin_specs.py`: Hook specifications
- `src/echemistpy/io/plugin_manager.py`: Plugin manager
- `src/echemistpy/io/plugings/echem/biologic_plugin.py`: BioLogic plugin wrapper
- `src/echemistpy/io/plugings/echem/lanhe_plugin.py`: LANHE plugin wrapper
- `src/echemistpy/io/plugings/generic_loaders.py`: Generic loader plugins
- `src/echemistpy/io/plugings/generic_savers.py`: Generic saver plugins
- `docs/PLUGIN_SYSTEM.md`: Comprehensive documentation

## Benefits

1. **Extensibility**: Easy to add new file formats
2. **Type Safety**: Traitlets provide parameter validation
3. **Separation of Concerns**: Clean plugin architecture
4. **Backward Compatibility**: Existing code continues to work
5. **Documentation**: Clear examples and API reference
6. **Testability**: Plugins can be tested independently

## Future Enhancements

Possible improvements:
1. Auto-discovery of plugins from entry points
2. Plugin priority/ordering configuration
3. Async plugin support for large files
4. Plugin validation on registration
5. More comprehensive error handling
6. Plugin dependency management
