# Changelog

## Unreleased

### Data Structures Refactoring (2025-11-19)

#### Type Renaming for Clarity
- **Breaking Change**: Renamed core data types for better clarity:
  - `RawMetadata` → `RawDataInfo` (metadata for raw data)
  - `MeasurementInfo` → `MeasurementMetadata` (metadata for standardized measurements)
  - `Results` → `AnalysisResult` (processed analysis results)
  - `ResultsInfo` → `AnalysisResultInfo` (metadata for analysis results)
- **Backward Compatibility**: Old names remain available as aliases, ensuring existing code continues to work

#### Enhanced Helper Methods
All six fundamental data types now include comprehensive helper methods:

**RawDataInfo**:
- `to_dict()` - Convert metadata to dictionary
- `get(key, default)` - Get metadata value by key
- `update(other)` - Update metadata with new values

**RawData**:
- `to_dict()` - Convert data to dictionary representation
- `get_variables()` - Get list of all variable names
- `get_coords()` - Get list of all coordinate names
- `select(variables)` - Select specific variables
- `__getitem__(key)` - Direct variable access via `data["variable_name"]`

**MeasurementMetadata**:
- `to_dict()` - Convert metadata to dictionary
- `get(key, default)` - Get metadata value (checks both standard fields and `others`)
- `update_others(metadata)` - Update the `others` dictionary

**Measurement**:
- `to_dict()` - Convert data to dictionary representation
- `get_variables()` - Get list of all variable names
- `get_coords()` - Get list of all coordinate names
- `select(variables)` - Select specific variables
- `__getitem__(key)` - Direct variable access

**AnalysisResultInfo**:
- `to_dict()` - Convert metadata to dictionary
- `get_parameter(key, default)` - Get analysis parameter by key
- `update_parameters(params)` - Update analysis parameters
- `add_remark(remark, separator)` - Append remarks with custom separator

**AnalysisResult**:
- `to_dict()` - Convert data to dictionary representation
- `get_variables()` - Get list of all variable names
- `get_coords()` - Get list of all coordinate names
- `select(variables)` - Select specific variables
- `__getitem__(key)` - Direct variable access

#### NX (NeXus) Structure Enhancements

**NXField**:
- `from_dataarray(name, dataarray, dtype)` - Create NXField from xarray.DataArray

**NXGroup**:
- `add_field(field)` - Add field with fluent interface (method chaining)
- `add_group(group)` - Add subgroup with fluent interface
- `add_link(link)` - Add link with fluent interface
- `from_dataset(name, dataset, nx_class)` - Create NXGroup from xarray.Dataset

#### Bug Fixes
- Fixed missing `import pandas as pd` in `loaders.py`
- Fixed incorrect field name `extras` → `others` in `MeasurementMetadata` instantiation

#### Testing
- Added comprehensive unit tests for all helper methods in `tests/io/test_structures_helpers.py`
- Tests cover all six data types, NX structures, and backward compatibility

#### Usage Examples

```python
from echemistpy.io.structures import RawData, MeasurementMetadata
import xarray as xr
import numpy as np

# Create and use RawData with helper methods
dataset = xr.Dataset({
    "voltage": (["row"], np.array([3.0, 3.5, 4.0])),
    "current": (["row"], np.array([0.1, 0.2, 0.3])),
}, coords={"row": np.arange(3)})

raw_data = RawData(data=dataset)
variables = raw_data.get_variables()  # ['voltage', 'current']
voltage = raw_data["voltage"]  # Direct access
data_dict = raw_data.to_dict()  # Convert to dict

# Use MeasurementMetadata with enhanced methods
info = MeasurementMetadata(
    technique="CV",
    sample_name="Sample1",
    others={"custom": "value"}
)
technique = info.get("technique")  # "CV"
custom = info.get("custom")  # "value" (from others)

# Backward compatibility - old names still work
from echemistpy.io.structures import RawMetadata, MeasurementInfo
assert RawMetadata is RawDataInfo  # True
assert MeasurementInfo is MeasurementMetadata  # True
```

---

### Previous Changes

- 标准化仓库目录结构，新增示例、测试与文档目录。
- 添加专有 LICENSE、贡献指南以及打包元数据文件。

### IO Module Optimization
- **Data Structures**: Implemented `RawMeasurement`, `Measurement` (Standardized), and `AnalysisResult` hierarchy in `structures.py`.
- **Loaders**: Updated all loaders in `loaders.py` to return `RawMeasurement` with full metadata.
- **Standardization**: Renamed `organization.py` to `standardized.py` and implemented `standardize_measurement` for technique-specific data standardization.
- **Saver**: Enhanced `saver.py` to support:
    - CSV: Saves 1D/2D tabular data with metadata headers.
    - HDF5/NetCDF: Saves full `Measurement` and `AnalysisResult` objects with metadata as attributes and groups.
- **Verification**: Added `tests/test_io.py` to verify the complete Read → Standardize → Save workflow.
- **Pipeline Verification**: Added `tests/test_verify_echem_pipeline.py` to verify the specific `Biologic_GPCL.mpr` loading, standardization, and saving pipeline.
