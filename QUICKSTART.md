# Quick Start Guide - echemistpy

Get started with echemistpy in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/liuchzzyy/echemistpy.git
cd echemistpy

# Install the package
pip install -e .
```

## Basic Usage

### 1. Import the package

```python
import echemistpy as ecp
import numpy as np
```

### 2. Use a characterization technique

```python
# Create an electrochemistry instance
echem = ecp.techniques.Electrochemistry()

# Load your data (example with CSV)
# data = echem.load_data('your_data.csv')

# Preprocess the data
# preprocessed = echem.preprocess(smooth=True)

# Analyze the data
# results = echem.analyze(analysis_type='basic')
```

### 3. Data Processing

```python
# Create some example data
data = np.random.randn(100) + 10

# Normalize data
normalized = ecp.utils.normalize(data, method='minmax')

# Smooth noisy data
noisy_data = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
smoothed = ecp.utils.smooth(noisy_data, window_length=11, method='savgol')

# Baseline correction
with_baseline = data + np.linspace(0, 5, 100)  # Add artificial baseline
corrected = ecp.utils.baseline_correction(with_baseline, method='polynomial')
```

### 4. Visualization

```python
import matplotlib.pyplot as plt

# Create line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig, ax = ecp.visualization.plot_line(
    x, y, 
    xlabel='Time (s)', 
    ylabel='Signal', 
    title='Example Data'
)
plt.show()

# Create heatmap
data_2d = np.random.randn(20, 30)
fig, ax = ecp.visualization.plot_heatmap(
    data_2d,
    xlabel='X Position',
    ylabel='Y Position',
    title='Heatmap Example',
    cmap='viridis'
)
plt.show()
```

## Available Techniques

```python
# Electrochemistry
echem = ecp.techniques.Electrochemistry()

# X-ray Diffraction
xrd = ecp.techniques.XRD()

# X-ray Photoelectron Spectroscopy
xps = ecp.techniques.XPS()

# X-ray Absorption Spectroscopy
xas = ecp.techniques.XAS()

# Transmission Electron Microscopy
tem = ecp.techniques.TEM()

# Scanning Electron Microscopy
sem = ecp.techniques.SEM()

# Scanning Transmission X-ray Microscopy
stxm = ecp.techniques.STXM()

# Thermogravimetric Analysis
tga = ecp.techniques.TGA()

# Electrochemical Quartz Crystal Microbalance
eqcm = ecp.techniques.EQCM()

# ICP-OES
icp = ecp.techniques.ICPOES()
```

## Working with Real Data

### Loading CSV Data

```python
# Electrochemistry data from CSV
echem = ecp.techniques.Electrochemistry()
data = echem.load_data('echem_data.csv', file_format='csv')
```

### Loading Excel Data

```python
# XRD data from Excel
xrd = ecp.techniques.XRD()
data = xrd.load_data('xrd_pattern.xlsx')
```

### Loading HDF5 Data

```python
# STXM data from HDF5
stxm = ecp.techniques.STXM()
data = stxm.load_data('stxm_stack.hdf5')
```

## Common Workflows

### Workflow 1: Process and Plot Electrochemistry Data

```python
import echemistpy as ecp
import numpy as np
import matplotlib.pyplot as plt

# Load data
echem = ecp.techniques.Electrochemistry()
# data = echem.load_data('cv_data.csv')

# For demonstration, create synthetic data
x = np.linspace(-0.5, 0.5, 500)
y = 10 * np.exp(-(x - 0.2)**2 / 0.01) - 5 * np.exp(-(x + 0.1)**2 / 0.02)
y += np.random.randn(500) * 0.5

# Smooth the data
y_smooth = ecp.utils.smooth(y, window_length=15, method='savgol')

# Plot
fig, ax = ecp.visualization.plot_line(
    x, y_smooth,
    xlabel='Potential (V vs. Reference)',
    ylabel='Current (μA)',
    title='Cyclic Voltammogram'
)
plt.show()
```

### Workflow 2: Normalize and Compare Multiple Datasets

```python
import echemistpy as ecp
import numpy as np
import matplotlib.pyplot as plt

# Create multiple datasets
datasets = [
    np.random.randn(100) * 2 + 10,
    np.random.randn(100) * 3 + 15,
    np.random.randn(100) * 1.5 + 8,
]

# Normalize all datasets
normalized = [ecp.utils.normalize(d, method='minmax') for d in datasets]

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for i, d in enumerate(datasets):
    ax1.plot(d, label=f'Dataset {i+1}')
ax1.set_title('Original Data')
ax1.legend()

for i, d in enumerate(normalized):
    ax2.plot(d, label=f'Dataset {i+1}')
ax2.set_title('Normalized Data')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Next Steps

- Read the full documentation in `README_PACKAGE.md`
- Check out examples in `examples/usage_example.py`
- Explore the API documentation in the source code docstrings
- Contribute your own techniques following `CONTRIBUTING.md`

## Getting Help

- Read the documentation files
- Check the examples directory
- Look at the source code (it's well documented!)
- Open an issue on GitHub

## Tips

1. All techniques follow the same pattern: `load_data() -> preprocess() -> analyze()`
2. Use type hints in your IDE for better code completion
3. Check docstrings for detailed parameter information
4. Data validation is built-in - use it to catch issues early
5. Visualization functions return (fig, ax) for further customization

Happy analyzing! 🚀
