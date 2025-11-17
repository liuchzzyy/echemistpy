# echemistpy

A Python package for electrochemistry and materials characterization analysis.

## Overview

`echemistpy` provides a comprehensive, extensible framework for analyzing data from various electrochemical and materials characterization techniques. The package is designed with modularity and extensibility in mind, making it easy to add new techniques or customize existing ones.

## Supported Techniques

- **Electrochemistry (Echem)**: Cyclic voltammetry, chronoamperometry, and other electrochemical methods
- **Electrochemical Quartz Crystal Microbalance (EQCM)**: Mass change measurements
- **X-ray Diffraction (XRD)**: Ex-situ and operando measurements
- **X-ray Photoelectron Spectroscopy (XPS/UPS)**: Surface characterization
- **X-ray Absorption Spectroscopy (XAS)**: XANES, EXAFS, and operando measurements
- **Transmission Electron Microscopy (TEM)**: TEM, TEM-EDS, TEM-EELS
- **Scanning Electron Microscopy (SEM)**: SEM and SEM-EDS
- **Scanning Transmission X-ray Microscopy (STXM)**: Chemical imaging
- **Thermogravimetric Analysis (TGA)**: Thermal decomposition
- **Inductively Coupled Plasma Optical Emission Spectrometry (ICP-OES)**: Elemental analysis

## Installation

### From source

```bash
git clone https://github.com/liuchzzyy/echemistpy.git
cd echemistpy
pip install -e .
```

### Using conda environment

```bash
conda env create -f environment.yml
conda activate txm
pip install -e .
```

## Quick Start

```python
import echemistpy as ecp

# Load and analyze electrochemistry data
echem = ecp.techniques.Electrochemistry()
data = echem.load_data('your_data.csv')
preprocessed = echem.preprocess()
results = echem.analyze()

# Load and analyze XRD data
xrd = ecp.techniques.XRD()
data = xrd.load_data('xrd_pattern.csv')
results = xrd.analyze()

# Use visualization utilities
from echemistpy.visualization import plot_line
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
fig, ax = ecp.visualization.plot_line(x, y, xlabel='X', ylabel='Y', title='Example Plot')
```

## Package Structure

```
echemistpy/
├── core/                  # Core base classes and exceptions
│   ├── base.py           # BaseCharacterization and BaseData classes
│   └── exceptions.py     # Custom exception classes
├── io/                   # Data loading and saving utilities
│   └── loaders.py        # File format loaders (CSV, Excel, HDF5, NetCDF)
├── techniques/           # Technique-specific implementations
│   ├── echem.py         # Electrochemistry
│   ├── eqcm.py          # EQCM
│   ├── xrd.py           # X-ray Diffraction
│   ├── xps.py           # X-ray Photoelectron Spectroscopy
│   ├── xas.py           # X-ray Absorption Spectroscopy
│   ├── tem.py           # Transmission Electron Microscopy
│   ├── sem.py           # Scanning Electron Microscopy
│   ├── stxm.py          # Scanning Transmission X-ray Microscopy
│   ├── tga.py           # Thermogravimetric Analysis
│   └── icp_oes.py       # ICP-OES
├── utils/                # Utility functions
│   ├── data_processing.py  # Data processing utilities
│   └── validation.py       # Data validation utilities
└── visualization/        # Plotting utilities
    └── plotting.py       # Common plotting functions
```

## Extensibility

The package is designed to be easily extensible. To add a new characterization technique:

1. Create a new class inheriting from `BaseCharacterization`
2. Implement the required methods: `load_data()`, `preprocess()`, and `analyze()`
3. Add technique-specific functionality

Example:

```python
from echemistpy.core.base import BaseCharacterization, BaseData
from pathlib import Path
from typing import Any

class NewTechnique(BaseCharacterization):
    def __init__(self):
        super().__init__("NewTechnique")
    
    def load_data(self, filepath: Path | str, **kwargs: Any) -> BaseData:
        # Implement data loading
        pass
    
    def preprocess(self, **kwargs: Any) -> BaseData:
        # Implement preprocessing
        pass
    
    def analyze(self, **kwargs: Any) -> dict[str, Any]:
        # Implement analysis
        pass
```

## Key Features

### Consistent API
All characterization techniques follow the same API pattern:
- `load_data()`: Load data from files
- `preprocess()`: Preprocess and clean data
- `analyze()`: Perform analysis and extract results

### Data Format Flexibility
The package supports multiple data formats:
- CSV
- Excel (xlsx, xls)
- HDF5
- NetCDF
- HyperSpy-compatible formats (for microscopy data)

### Comprehensive Utilities
- Data normalization (min-max, z-score, L2)
- Smoothing (Savitzky-Golay, moving average)
- Baseline correction
- Data validation
- Common plotting functions

### Type Hints and Documentation
All functions include type hints and comprehensive docstrings for better code completion and documentation.

## Dependencies

Main dependencies include:
- numpy
- pandas
- xarray
- matplotlib
- scipy
- hyperspy (for microscopy data)
- h5py (for HDF5 files)

See `pyproject.toml` or `environment.yml` for the complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Author

Cheng Liu, PhD

## Citation

If you use this package in your research, please cite:

```
Liu, C. (2024). echemistpy: A Python package for electrochemistry and materials characterization analysis.
```
