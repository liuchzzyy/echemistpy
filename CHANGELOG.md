# Changelog

## Unreleased
- 标准化仓库目录结构，新增示例、测试与文档目录。
- 添加专有 LICENSE、贡献指南以及打包元数据文件。

### IO Module Optimization
- **Data Structures**: Implemented `RawMeasurement`, `Measurement` (Standardized), and `Results` hierarchy in `structures.py`.
- **Loaders**: Updated all loaders in `loaders.py` to return `RawMeasurement` with full metadata.
- **Standardization**: Renamed `organization.py` to `standardized.py` and implemented `standardize_measurement` for technique-specific data standardization.
- **Saver**: Enhanced `saver.py` to support:
    - CSV: Saves 1D/2D tabular data with metadata headers.
    - HDF5/NetCDF: Saves full `Measurement` and `Results` objects with metadata as attributes and groups.
- **Verification**: Added `tests/test_io.py` to verify the complete Read -> Standardize -> Save workflow.
- **Pipeline Verification**: Added `tests/test_verify_echem_pipeline.py` to verify the specific `Biologic_GPCL.mpr` loading, standardization, and saving pipeline.
