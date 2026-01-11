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
│   ├── analysis/             # Domain Logic (See src/echemistpy/analysis/AGENTS.md)
│   │   ├── xas/              # X-ray Absorption Spectroscopy
│   │   ├── echem/            # Electrochemistry
│   │   └── stxm/             # Scanning Transmission X-ray Microscopy
│   └── ...
└── tests/                    # Pytest suite
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| **Add Instrument Support** | `src/echemistpy/io/plugins/` | Implement `BaseReader` subclass |
| **Add Analysis Algorithm** | `src/echemistpy/analysis/{domain}/` | Implement `TechniqueAnalyzer` subclass |
| **Data Structure Logic** | `src/echemistpy/io/structures.py` | `RawData` / `AnalysisData` definitions |
| **XAS Logic** | `src/echemistpy/analysis/xas/` | Self-contained domain module |

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
- **Domain-Driven Design**: Logic is grouped by scientific domain (`xas`, `echem`) not software layer.
