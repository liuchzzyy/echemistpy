# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**echemistpy** is a unified data processing framework for electrochemical techniques and materials characterization (XRD, XPS, TGA, XAS, TXM). It uses a scalable reader + analyzer pattern with pipeline orchestration architecture built on xarray, numpy, and scipy.

### Key Features
- Unified data format using `xarray.Dataset` (flat data) and `xarray.DataTree` (hierarchical data)
- Extensible plugin-based reader interface for different instruments/formats
- Modular analyzer templates with `TechniqueAnalyzer` base class
- Pipeline orchestration for batch processing with `AnalysisPipeline`
- Type-safe configuration using traitlets for metadata management
- Plugin registration mechanism using pluggy

## Development Commands

### Environment Setup (uv)
```powershell
# Install core dependencies and create virtual environment
uv sync

# Install development tools (ruff, pytest, pre-commit)
uv sync --only-group dev

# Install documentation tools
uv sync --only-group docs

# Install Jupyter/interactive tools
uv sync --only-group interactive

# Install all dependency groups
uv sync --all-groups
```

### Code Quality
```powershell
# Lint and format
uv run ruff check src/
uv run ruff format src/

# Type checking
uv run ty check

# Run tests
uv run pytest
```

### Git Commit Conventions
- **Language**: All commit messages MUST be in **Chinese**
- **Format**: Conventional commits with prefixes: `[FEATURE]`, `[FIX]`, `[DOCS]`, etc.
- **Example**: `[FEATURE] 添加库伦效率计算功能`

## Architecture

### Core Data Flow

```
Raw Data Files → IOPluginManager → RawData + RawDataInfo
                                        ↓
                                 TechniqueAnalyzer
                                        ↓
                                ResultsData + ResultsDataInfo
```

### Key Components

**Data Structures** (`src/echemistpy/io/structures.py`)
- `RawData` + `RawDataInfo`: Container for measurement data and metadata
- `ResultsData` + `ResultsDataInfo`: Container for processed results
- Backend: `xarray.Dataset` for flat data, `xarray.DataTree` for hierarchical data (e.g., XRD scans at different temperatures)
- Metadata: Uses traitlets for validation and dynamic parameter storage

**I/O System** (`src/echemistpy/io/`)
- `load()`: Unified loading interface with auto-format detection
- `IOPluginManager`: Plugin-based system for different instruments
- `DataStandardizer`: Normalizes column names, units, and time formats
- Supports both files and directories as input

**Processing System** (`src/echemistpy/processing/`)
- `TechniqueAnalyzer`: Abstract base class for all analyzers (implements `compute()`, optional `validate()` and `preprocess()`)
- `TechniqueRegistry`: Maps technique types to analyzer implementations
- `AnalysisPipeline`: Orchestrates batch processing workflows across multiple samples

### Supported Instruments & Formats

**Electrochemistry**
- BioLogic (.mpt) - `BiologicMPTReader`
- LANHE (.xlsx) - `LanheXLSXReader`

**Materials Characterization**
- XRD: MSPD (.xye) - `MSPDReader`
- XAS: CLAESS (.dat) - `CLAESSReader`
- TXM: MISTRAL (.hdf5) - `MistralHDF5Reader`

## Common Patterns

### Data Loading
```python
from echemistpy.io import load

# Auto-detect format from file extension
raw_data, raw_info = load("data.mpt", sample_name="MySample")

# Explicit instrument for ambiguous formats (e.g., .xlsx)
raw_data, raw_info = load("data.xlsx", instrument="lanhe")
```

### Data Standardization Conventions

All readers output standardized column names via `DataStandardizer`:

**Time Columns:**
- `systime`: Absolute time (datetime64[ns])
- `time_s`: Relative time in seconds (float64)

**Electrochemistry Columns:**
- `Ewe/V`: Working electrode potential (volts)
- `<I>/mA`: Current (milliamps)

**Coordinate Names:**
- Use `"record"` or `"row"` as primary dimension for tabular data

### Working with DataTree (Hierarchical Data)

```python
from echemistpy.io import load

raw_data, raw_info = load("xrd_data.xye")

# Check if data is hierarchical
if raw_data.is_tree:
    tree = raw_data.data  # xarray.DataTree
    # Access specific node (e.g., scan at specific temperature)
    scan_data = tree["scan_1"].to_dataset()
```

### Implementing Analyzers

```python
from echemistpy.processing.analyzers.base import TechniqueAnalyzer

class MyAnalyzer(TechniqueAnalyzer):
    technique = "my_technique"  # Register in TechniqueRegistry
    required_columns = ("Ewe/V", "<I>/mA")  # Validate input data

    def compute(self, data, **kwargs):
        # Main calculation logic
        summary = {"mean": data.mean().values}
        results_table = {"processed": data.to_dataframe()}
        return summary, results_table

    def validate(self, data):
        # Optional: Validate data before processing
        super().validate(data)

    def preprocess(self, data):
        # Optional: Preprocess data (e.g., filtering, smoothing)
        return data
```

## Extension Points

### Adding New Readers

1. Create plugin class in `src/echemistpy/io/plugins/[technique]/` directory
2. Implement `load()` method returning `(RawData, RawDataInfo)`
3. Register in `IOPluginManager` (`src/echemistpy/io/plugin_manager.py`) with `supported_extensions` and `instrument` name
4. **Naming**: Follow PEP 8, but instrument-specific files (e.g., `MSPD.py`) may use original names with `# ruff: noqa: N999`

### Adding New Analyzers

1. Inherit from `TechniqueAnalyzer` in `src/echemistpy/processing/analyzers/`
2. Implement `compute()` method returning `(summary_dict, results_table_dict)`
3. Optionally override `validate()` and `preprocess()`
4. Set `technique` class attribute and `required_columns` tuple
5. Register in `TechniqueRegistry` or use custom registry

## Code Quality Standards

**Ruff Configuration:**
- Line length: 200
- Extensive linting: Pyflakes (F), PEP 8 (E/W), flake8-bugbear (B), pep8-naming (N), pylint (PL), security (S)
- Auto-fix imports with isort (I)
- Per-file ignores for tests: `S101` (assert usage), `ARG` (unused args)

**Ty Configuration:**
- Strict type checking required
- Focus: `RawData.data` attribute must correctly type `Dataset` vs `DataTree`

## Project Structure

```
src/echemistpy/
├── io/                      # Data structures & I/O
│   ├── structures.py        # RawData, ResultsData, traitlets configs
│   ├── plugin_manager.py    # IOPluginManager with pluggy hooks
│   ├── loaders.py           # Unified load() interface
│   └── plugins/             # Reader implementations by technique
│       ├── echem/           # Electrochemistry readers
│       ├── xrd/             # XRD readers
│       └── ...
├── processing/              # Analysis & preprocessing
│   ├── analyzers/           # TechniqueAnalyzer implementations
│   └── analysis/            # Analysis-specific modules
├── pipelines/               # Orchestration (AnalysisPipeline)
└── utils/                   # Utilities (visualization, helpers)
```

## Important Notes

- Project status: **Alpha** - Design may change between versions
- Metadata uses traitlets for runtime validation and type safety
- Plugin system uses pluggy for flexible extension registration
- Public APIs use English; internal code uses Chinese comments
- Report issues on GitHub Issues tracker
- Environment name in Conda: `echemistpy` (defined in `environment.yml`)
