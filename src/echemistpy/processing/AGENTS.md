# PROCESSING MODULE KNOWLEDGE BASE

## OVERVIEW
Provides a unified analysis pipeline. Analyzers are registered by technique and instrument, allowing generic calls like "analyze this XAS data".

## ARCHITECTURE
`TechniqueRegistry` -> `TechniqueAnalyzer` -> `analyze()` -> `_compute()`

## EXTENSION GUIDE
To add a new analysis (e.g., `NewMethod`):

1.  **Create File**: `src/echemistpy/processing/analyzers/new_method.py`
2.  **Inherit**: `from .registry import TechniqueAnalyzer`
3.  **Define**:
    - `technique = "new_method"`
    - `required_columns = ("col1", "col2")`
4.  **Implement**:
    - `_compute(raw_data)`: Return `(AnalysisData, AnalysisDataInfo)`
5.  **Register**: Add to `create_default_registry()` in `registry.py`.

## KEY CLASSES
- **`TechniqueAnalyzer`**: Base class. Handles validation, preprocessing, and metadata inheritance.
- **`TechniqueRegistry`**: Stores available analyzers.
- **`AnalysisData`**: Container for results (e.g., fitted curves, calculated metrics).

## CONVENTIONS
- **Do Not Override `analyze()`**: Override `_compute()` instead.
- **Metadata**: Standard metadata (`sample_name`, `operator`) is automatically copied from input to output.
- **State**: Analyzers should be stateless regarding data (don't store data in `self`).
- **XAS**: Specialized logic in `analyzers/xas.py` uses `xraylarch`.
