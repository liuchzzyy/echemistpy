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

- Unified data model: represent experimental data as `xarray.Dataset`
- Reader interfaces: extensible loaders and standardization for instruments/formats
- Modular analyzers: template-based analyzers, easy to extend for new techniques
- Pipeline orchestration: batch processing with automatic summary aggregation
- Type-safe configuration: traitlets-based validation and consistency
- Plugin architecture: pluggy registry for flexible technique support

> Note: echemistpy is under active development. Designs may evolve. Please report issues via the [Issue Tracker](https://github.com/liuchzzyy/echemistpy/issues).

---

## Quick Start

### Install (uv recommended)

```powershell
# Optional: install target Python (if you need a specific version)
uv python install 3.11

# Sync dependencies and auto-create/update virtualenv based on pyproject.toml
uv sync

# Activate default virtual environment (.venv)
.venv\Scripts\activate
```

### Optional dependency groups (install as needed)

```powershell
# Development tools (ruff, pytest, pre-commit)
uv sync --only-group dev

# Documentation
uv sync --only-group docs

# Jupyter interactive
uv sync --only-group interactive

# All groups
uv sync --all-groups
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

---

## Project Structure

```
echemistpy/
├── src/echemistpy/
│   ├── io/              # Data structures & I/O
│   ├── processing/      # Analysis & preprocessing
│   │   └── analysis/    # Analyzer implementations
│   ├── pipelines/       # Pipeline orchestration
│   └── utils/           # Readers & visualization
├── tests/               # Unit tests
├── examples/            # Example data
├── docs/                # Jupyter Notebooks
├── pyproject.toml       # Project configuration
└── environment.yml      # Conda environment (optional)
```

---

## Contributing

We welcome issues and pull requests!

### Development Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes with tags: `[FEATURE]`, `[FIX]`, `[DOCS]`, etc.
4. Run tests: `pytest tests/`
5. Check code quality: `ruff check src/`
6. Open a Pull Request

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

---
