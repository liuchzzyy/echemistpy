# Contributing to echemistpy

Thank you for your interest in contributing to echemistpy! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/echemistpy.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install in development mode: `pip install -e .`

## Development Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Run linter
ruff check echemistpy/

# Run formatter
ruff format echemistpy/
```

## Adding a New Technique

To add a new characterization technique:

1. Create a new file in `echemistpy/techniques/` (e.g., `new_technique.py`)
2. Create a class that inherits from `BaseCharacterization`
3. Implement the required methods: `load_data()`, `preprocess()`, and `analyze()`
4. Add the new technique to `echemistpy/techniques/__init__.py`
5. Add documentation and examples

Example:

```python
from echemistpy.core.base import BaseCharacterization, BaseData
from pathlib import Path
from typing import Any

class NewTechnique(BaseCharacterization):
    def __init__(self):
        super().__init__("NewTechnique")
    
    def load_data(self, filepath: Path | str, **kwargs: Any) -> BaseData:
        # Implementation
        pass
    
    def preprocess(self, **kwargs: Any) -> BaseData:
        # Implementation
        pass
    
    def analyze(self, **kwargs: Any) -> dict[str, Any]:
        # Implementation
        pass
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings in NumPy style
- Keep functions focused and modular
- Use meaningful variable names

## Testing

- Add tests for new functionality
- Ensure existing tests pass
- Test with different Python versions (3.10+)

## Documentation

- Update README.md if adding major features
- Add docstrings to all public functions and classes
- Update examples if needed

## Pull Request Process

1. Ensure your code passes linting: `ruff check echemistpy/`
2. Update documentation as needed
3. Commit your changes with clear messages
4. Push to your fork
5. Submit a pull request to the main repository

## Questions?

Feel free to open an issue if you have questions or need help!
