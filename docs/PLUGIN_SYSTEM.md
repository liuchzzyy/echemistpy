# Plugin System Documentation

## Overview

The echemistpy io module now uses a pluggy-based plugin system for extensible file loading and saving. This allows users to easily add support for new file formats by creating custom plugins.

## Architecture

### Key Components

1. **Plugin Specifications** (`plugin_specs.py`): Defines the hook specifications that plugins must implement
2. **Plugin Manager** (`plugin_manager.py`): Manages plugin registration and dispatching
3. **Loader Plugins**: Plugins that load data from files
4. **Saver Plugins**: Plugins that save data to files

### Built-in Plugins

#### Echem-Specific Plugins
- **BiologicLoaderPlugin**: Loads BioLogic MPT/MPR files
- **LanheLoaderPlugin**: Loads LANHE Excel files

#### Generic Format Plugins
- **CSVLoaderPlugin**: Loads CSV/TSV files
- **ExcelLoaderPlugin**: Loads Excel files (xlsx, xls)
- **HDF5LoaderPlugin**: Loads HDF5/NetCDF files
- **CSVSaverPlugin**: Saves to CSV/TSV files
- **HDF5SaverPlugin**: Saves to HDF5/NetCDF files
- **JSONSaverPlugin**: Saves to JSON files

## Using the Plugin System

### Loading Data

```python
from echemistpy.io import load_data_file

# Load a file - plugin is selected automatically based on extension
raw_data, raw_data_info = load_data_file("data.mpt")

# Or specify format explicitly
raw_data, raw_data_info = load_data_file("data.txt", fmt="csv")
```

### Saving Data

```python
from echemistpy.io import save_measurement

# Save measurement - plugin is selected based on extension
save_measurement(measurement, measurement_info, "output.csv")

# Or specify format explicitly
save_measurement(measurement, measurement_info, "output.h5", fmt="hdf5")
```

### Getting Plugin Information

```python
from echemistpy.io import get_plugin_manager, list_supported_formats

# Get plugin manager instance
pm = get_plugin_manager()

# List supported loaders
print(pm.get_supported_loaders())
# {'mpt': 'BiologicLoaderPlugin', 'csv': 'CSVLoaderPlugin', ...}

# List supported savers
print(pm.get_supported_savers())
# {'csv': 'CSVSaverPlugin', 'json': 'JSONSaverPlugin', ...}

# Or use the convenience function
print(list_supported_formats())
```

## Creating Custom Plugins

### Loader Plugin Example

```python
from pathlib import Path
from typing import Any
import xarray as xr
from traitlets import HasTraits, Unicode
from echemistpy.io.plugin_specs import hookimpl
from echemistpy.io.structures import RawData, RawDataInfo
from echemistpy.io import register_loader

class MyCustomLoaderPlugin(HasTraits):
    """Custom loader plugin with traitlets parameter control."""
    
    # Traitlets properties for configuration
    encoding = Unicode(default_value="utf-8", help="File encoding").tag(config=True)
    
    @hookimpl
    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return ["xyz", "custom"]
    
    @hookimpl
    def can_load(self, filepath: Path) -> bool:
        """Check if this loader can handle the given file."""
        return filepath.suffix.lower() in [".xyz", ".custom"]
    
    @hookimpl
    def load_file(self, filepath: Path, **kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """Load a custom file format.
        
        Args:
            filepath: Path to the file
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        # Your custom loading logic here
        # Example: read file and create xarray.Dataset
        import numpy as np
        
        # Create some sample data
        data_dict = {
            "column1": ("row", np.array([1, 2, 3])),
            "column2": ("row", np.array([4, 5, 6])),
        }
        dataset = xr.Dataset(data_dict, coords={"row": np.arange(3)})
        
        # Create metadata
        metadata = {
            "technique": "Custom",
            "filename": filepath.stem,
            "extension": filepath.suffix,
        }
        
        raw_data = RawData(data=dataset)
        raw_data_info = RawDataInfo(meta=metadata)
        
        return raw_data, raw_data_info

# Register the plugin
register_loader(MyCustomLoaderPlugin(), name="my_custom_loader")

# Now you can use it
from echemistpy.io import load_data_file
raw_data, raw_data_info = load_data_file("data.xyz")
```

### Saver Plugin Example

```python
from pathlib import Path
from typing import Any
import xarray as xr
from traitlets import HasTraits
from echemistpy.io.plugin_specs import hookimpl
from echemistpy.io import get_plugin_manager

class MyCustomSaverPlugin(HasTraits):
    """Custom saver plugin."""
    
    @hookimpl
    def get_supported_formats(self) -> list[str]:
        """Return list of supported output formats."""
        return ["xyz", "custom"]
    
    @hookimpl
    def save_data(
        self,
        data: xr.Dataset,
        metadata: dict[str, Any],
        filepath: Path,
        fmt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save data to custom format.
        
        Args:
            data: xarray.Dataset to save
            metadata: Metadata dictionary
            filepath: Destination path
            fmt: Optional format override
            **kwargs: Additional parameters
        """
        # Your custom saving logic here
        with open(filepath, "w") as f:
            f.write("# Custom file format\n")
            # Write metadata
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            # Write data
            for var_name in data.data_vars:
                f.write(f"{var_name}: {data[var_name].values.tolist()}\n")

# Register the plugin
pm = get_plugin_manager()
pm.register_plugin(MyCustomSaverPlugin(), name="my_custom_saver")

# Now you can use it
from echemistpy.io import save_measurement
save_measurement(measurement, measurement_info, "output.xyz")
```

## Using Traitlets for Configuration

Traitlets provide type validation and configuration management for plugin parameters:

```python
from traitlets import HasTraits, Unicode, Integer, Bool

class ConfigurableLoaderPlugin(HasTraits):
    """Loader with configurable parameters."""
    
    # String parameter with default and help text
    encoding = Unicode(default_value="utf-8", help="File encoding").tag(config=True)
    
    # Integer parameter with validation
    max_rows = Integer(default_value=1000, help="Maximum rows to read").tag(config=True)
    
    # Boolean flag
    skip_errors = Bool(default_value=False, help="Skip errors during loading").tag(config=True)
    
    @hookimpl
    def load_file(self, filepath: Path, **kwargs: Any):
        # Use the configured parameters
        encoding = kwargs.get("encoding", self.encoding)
        max_rows = kwargs.get("max_rows", self.max_rows)
        skip_errors = kwargs.get("skip_errors", self.skip_errors)
        
        # Your loading logic using these parameters
        ...

# Create instance with custom configuration
loader = ConfigurableLoaderPlugin(encoding="latin1", max_rows=500)
```

## Plugin Priority

When multiple plugins support the same file extension, the last registered plugin takes precedence. The default plugins are registered in this order:

1. Echem-specific plugins (BiologicLoaderPlugin, LanheLoaderPlugin)
2. Generic format plugins (CSVLoaderPlugin, ExcelLoaderPlugin, HDF5LoaderPlugin)

For savers:
1. Generic saver plugins (CSVSaverPlugin, HDF5SaverPlugin, JSONSaverPlugin)

## Best Practices

1. **Use Traitlets for Parameters**: Leverage traitlets for type validation and configuration
2. **Implement All Required Hooks**: Ensure your plugin implements all necessary hookimpl methods
3. **Handle Errors Gracefully**: Return meaningful error messages when loading/saving fails
4. **Document Your Plugin**: Provide clear docstrings for your plugin class and methods
5. **Test Thoroughly**: Test your plugin with various file formats and edge cases

## Advanced Usage

### Unregistering Plugins

```python
from echemistpy.io import get_plugin_manager

pm = get_plugin_manager()

# Unregister a plugin
pm.unregister_plugin(my_plugin_instance)
```

### Checking Plugin Capabilities

```python
from echemistpy.io import get_plugin_manager

pm = get_plugin_manager()

# Check if a format is supported
if "xyz" in pm.get_supported_loaders():
    print("XYZ format is supported")
```
