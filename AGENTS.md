# PROJECT KNOWLEDGE BASE

**Project:** echemistpy
**Focus:** Electrochemical & Materials Data Analysis (XAS, XRD, Echem)

## OVERVIEW
Unified data processing library for scientific workflows. Standardizes data ingestion via plugins and orchestrates analysis via a registry pattern. Heavily relies on `xarray` for data containers and `traitlets` for metadata validation.

## STRUCTURE
```
.
├── scripts/                  # Workflow scripts (e.g., operando analysis)
├── src/echemistpy/
│   ├── core/                 # Shared constants and base types
│   ├── io/                   # Data ingestion (See src/echemistpy/io/AGENTS.md)
│   ├── processing/           # Analysis logic (See src/echemistpy/processing/AGENTS.md)
│   └── visualization/        # Plotting utilities (plot_xas.py)
└── tests/                    # Pytest suite
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| **Add Instrument Support** | `src/echemistpy/io/plugins/` | Implement `BaseReader` subclass |
| **Add Analysis Algorithm** | `src/echemistpy/processing/analyzers/` | Implement `TechniqueAnalyzer` subclass |
| **Data Structure Logic** | `src/echemistpy/io/structures.py` | `RawData` / `AnalysisData` definitions |
| **XAS Logic** | `src/echemistpy/processing/**/xas.py` | Distributed across preprocessing/analyzers |

## CONVENTIONS
- **Strict Typing**: All functions must have type hints. Use `uv run ty check`.
- **Data/Info Separation**:
  - `RawData` = `xarray` (Numerical data)
  - `RawDataInfo` = `traitlets` (Metadata)
- **Mixins**: Use `XarrayDataMixin` for common data operations.
- **Git**: Commit messages must be in **Chinese** with prefixes (`[FEATURE]`, `[FIX]`).

## COMMANDS
```bash
# Setup
uv sync --all-groups

# Quality
uv run ruff format src/
uv run ruff check src/ --fix
uv run ty check

# Test
uv run pytest
```

## ARCHITECTURE HIGHLIGHTS
- **IOPluginManager**: Singleton that auto-discovers readers in `io/plugins`.
- **TechniqueRegistry**: Maps `(technique, instrument)` to analyzers.
- **Metadata Inheritance**: Automatic flow of metadata from Raw -> Analysis results.
