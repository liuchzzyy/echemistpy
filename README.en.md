<h1 align="center">
  <br>
  <img src="docs/static/liuchzzyy.jpg" alt="echemistpy" width="150">
  <br>
  echemistpy
  <br>
</h1>

<p align="center">
<strong>Unified data processing for electrochemistry and materials characterization</strong><br/>
Cross-platform: Windows / macOS / Linux
</p>

<p align="center">
  <a href="https://cecill.info/licences/Licence_CeCILL-B_V1-en.html"><img src="https://img.shields.io/badge/License-CeCILL--B-blue.svg" alt="License: CeCILL-B"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue" alt="Python Version"></a>
  <a href="https://github.com/liuchzzyy/echemistpy"><img src="https://img.shields.io/badge/GitHub-echemistpy-black?logo=github" alt="GitHub"></a>
  <a href="https://github.com/liuchzzyy/echemistpy/issues"><img src="https://img.shields.io/github/issues/liuchzzyy/echemistpy" alt="Issues"></a>
</p>

---

<p align="center">
  <a href="README.md">中文说明</a> | <a href="README.en.md">README</a>
</p>

---

## What is echemistpy?

**echemistpy** is a unified data processing framework for electrochemical techniques and materials characterization. It uses an extensible Reader + Analyzer pattern with pipeline orchestration, built on xarray, numpy, scipy, and pluggy.

### Key Features

- **Unified Data Model**: Represent experimental data as `xarray.Dataset` (flat) and `xarray.DataTree` (hierarchical)
- **Reader Interfaces**: Extensible loaders and standardization for various instruments/formats
- **Modular Analyzers**: Template-based analyzers, easy to extend for new techniques
- **Pipeline Orchestration**: Batch processing with automatic summary aggregation
- **Type-safe Configuration**: Traitlets-based validation and consistency
- **Plugin Architecture**: Pluggy registry for flexible technique support

> **Note**: echemistpy is under active development. Designs may evolve. Please report issues via the [Issue Tracker](https://github.com/liuchzzyy/echemistpy/issues).

---

## Quick Start

### Install (uv recommended)

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```powershell
# 1. Sync dependencies (auto-creates/updates virtualenv)
uv sync

# 2. Install all optional groups (docs, interactive tools, etc.)
uv sync --all-groups

# 3. Activate environment
.venv\Scripts\activate
```

### Development Workflow

Before committing, please run the following quality checks:

```powershell
# Format code
uv run ruff format src/

# Lint code
uv run ruff check src/ --fix

# Type checking
uv run ty check

# Run tests
uv run pytest
```

---

## Usage Examples

### Data Loading & Standardization (I/O)

`echemistpy` provides a unified `load` interface that automatically detects file formats and standardizes them with consistent column names and units.

```python
from echemistpy.io import load

# Load a BioLogic .mpt file
raw_data, raw_info = load("docs/examples/echem/Biologic_GPCL.mpt", sample_name="MySample")

# Explore standardized data (xarray.Dataset)
print(raw_data.data)

# Access metadata
print(raw_info.to_dict())
```

### Analyzing Data

```python
from echemistpy.processing.analyzers.echem import GalvanostaticAnalyzer

# Initialize analyzer
analyzer = GalvanostaticAnalyzer()

# Run analysis
result_data, result_info = analyzer.analyze(raw_data)
```

---

## Architecture

```
Raw Files -> IOPluginManager -> RawData + RawDataInfo
                                       |
                                TechniqueAnalyzer.analyze()
                                       |
                               AnalysisData + AnalysisDataInfo
```

- **IOPluginManager**: Auto-detects formats and dispatches to appropriate `Reader`
- **RawData**: Stores raw measurement data (based on `xarray`)
- **TechniqueAnalyzer**: Base class for analysis algorithms (validation -> preprocessing -> computation)
- **AnalysisData**: Stores analysis results with a unified interface

---

## Project Structure

```
echemistpy/
├── src/echemistpy/
│   ├── io/                  # Core I/O system
│   │   ├── plugins/         # Instrument/format reader plugins
│   │   ├── loaders.py       # Unified load() interface
│   │   ├── structures.py    # Data structures (RawData, AnalysisData)
│   │   └── plugin_manager.py # Plugin manager
│   ├── processing/          # Data processing
│   │   ├── analyzers/       # Analysis algorithms by technique
│   │   └── pipeline.py      # Analysis pipeline orchestration
│   └── utils/               # Utilities
├── tests/                   # Unit tests
├── docs/                    # Documentation & Jupyter Notebooks
└── pyproject.toml           # Project configuration
```

---

## Contributing

We welcome issues and pull requests!

### Development Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes (Please use **Chinese** commit messages with tags: `[FEATURE]`, `[FIX]`, `[DOCS]`, etc.)
4. Run full checks: format, lint, type check, test
5. Open a Pull Request

See [AGENTS.md](AGENTS.md) for detailed development guidelines.

---

## License

[CeCILL-B Free Software License Agreement](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)

For details, see the [LICENSE](LICENSE) file in the repository.

---

## Citation

```bibtex
@software{echemistpy,
  author = {Cheng Liu},
  title = {echemistpy},
  url = {https://github.com/liuchzzyy/echemistpy},
  year = {2025}
}
```

**Last Updated**: Jan 7, 2026
