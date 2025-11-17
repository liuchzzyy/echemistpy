# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-17

### Added
- Initial package structure with modular architecture
- Core module with `BaseCharacterization` and `BaseData` classes
- Custom exception classes for error handling
- IO module with support for CSV, Excel, HDF5, and NetCDF formats
- Utility modules for data processing and validation
  - Normalization (min-max, z-score, L2)
  - Smoothing (Savitzky-Golay, moving average)
  - Baseline correction (polynomial, linear)
  - Data validation functions
- Visualization module with common plotting functions
  - Line plots
  - Heatmaps
  - Contour plots
- Technique-specific implementations:
  - Electrochemistry (Echem)
  - Electrochemical Quartz Crystal Microbalance (EQCM)
  - X-ray Diffraction (XRD)
  - X-ray Photoelectron Spectroscopy (XPS)
  - X-ray Absorption Spectroscopy (XAS)
  - Transmission Electron Microscopy (TEM)
  - Scanning Electron Microscopy (SEM)
  - Scanning Transmission X-ray Microscopy (STXM)
  - Thermogravimetric Analysis (TGA)
  - Inductively Coupled Plasma Optical Emission Spectrometry (ICP-OES)
- Comprehensive documentation in README_PACKAGE.md
- Usage examples in examples/usage_example.py
- MIT License
- pyproject.toml with modern Python packaging
- MANIFEST.in for package distribution
- .gitignore for Python projects

### Features
- Extensible architecture using abstract base classes
- Consistent API across all characterization techniques
- Type hints throughout the codebase
- NumPy-style docstrings
- Support for multiple data formats
- Data conversion utilities (DataFrame, xarray)

[0.1.0]: https://github.com/liuchzzyy/echemistpy/releases/tag/v0.1.0
