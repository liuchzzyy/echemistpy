# AGENTS.md

Guidelines for AI agents working on the echemistpy codebase.

## Build, Lint, and Test Commands

All commands use `uv` as the package manager. Run from the project root.

### Environment Setup

```powershell
uv sync                    # Install core dependencies
uv sync --all-groups       # Install all dependency groups (dev, docs, interactive, test)
```

### Linting and Formatting

```powershell
uv run ruff check src/           # Lint code
uv run ruff check src/ --fix     # Lint and auto-fix
uv run ruff format src/          # Format code
```

### Type Checking

```powershell
uv run ty check                  # Run type checker (strict mode)
```

### Testing

```powershell
uv run pytest                    # Run all tests
uv run pytest --cov=echemistpy   # Run with coverage

# Run a single test file
uv run pytest tests/integration/test_io_with_real_data.py

# Run a specific test function
uv run pytest tests/integration/test_io_with_real_data.py::test_biologic_gpcl

# Run tests matching a pattern
uv run pytest -k "biologic"
```

## Code Style Guidelines

### Import Organization

Organize imports in 4 groups, separated by blank lines:

```python
from __future__ import annotations          # 1. Future imports (always first)

import logging                               # 2. Standard library
from pathlib import Path
from typing import Any, ClassVar

import numpy as np                           # 3. Third-party packages
import xarray as xr
from traitlets import HasTraits, Unicode

from echemistpy.io.structures import RawData  # 4. Local imports
```

- Use `isort` ordering (handled by `ruff` with `I` rule)
- Use `from traitlets import List as TList` to avoid shadowing built-in `list`

### Formatting

- **Line length**: 200 characters (configured in `pyproject.toml`)
- **Formatter**: ruff format
- **Pre-commit hooks**: trailing whitespace, end-of-file fixer, mixed line endings

### Type Annotations

- Always use type annotations for function parameters and return values
- Prefer modern syntax: `list[str]`, `dict[str, Any]`, `str | None`
- Use `ClassVar` for class-level constants: `INSTRUMENT_NAME: ClassVar[str] = "BioLogic"`
- Use `TypeVar` for generic types: `T = TypeVar("T", bound="BaseInfo")`

```python
def load(path: str | Path, instrument: str | None = None) -> tuple[RawData, RawDataInfo]:
    ...

class MyReader:
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".mpt", ".txt"}
```

### Naming Conventions

| Type | Convention | Examples |
|------|------------|----------|
| Classes | PascalCase | `BiologicMPTReader`, `TechniqueAnalyzer` |
| Functions/Methods | snake_case | `load_single_file()`, `standardize_names()` |
| Private | Leading underscore | `_parse_metadata()`, `_internal_helper()` |
| Constants | UPPER_SNAKE_CASE | `BOOL_COLUMNS`, `ECHEM_PREFERRED_ORDER` |
| Modules | lowercase | `structures.py`, `loaders.py` |

Class suffixes by role:
- `*Reader` - I/O plugins
- `*Analyzer` - Processing analyzers
- `*Data` / `*Info` - Data containers
- `*Mixin` - Shared functionality

### Docstrings

Use Google-style docstrings. Public APIs use English; internal code may use Chinese.

```python
def compute(self, raw_data: RawData) -> tuple[AnalysisData, AnalysisDataInfo]:
    """Compute analysis results from raw data.

    Args:
        raw_data: Input measurement data with required columns.

    Returns:
        Tuple of (AnalysisData, AnalysisDataInfo) containing results.

    Raises:
        ValueError: If required columns are missing.
    """
```

### Error Handling

- Catch specific exceptions first, generic `Exception` last
- Include context in error messages: `raise ValueError(f"No loader for '{ext}'")`
- Use logging for non-critical errors: `logger.warning("...", exc_info=True)`
- Use `warnings.warn(..., stacklevel=2)` for deprecation or non-fatal issues

```python
try:
    result = process_file(path)
except FileNotFoundError:
    raise ValueError(f"Data file not found: {path}") from None
except (OSError, TypeError) as e:
    logger.warning("Processing failed for %s: %s", path, e)
```

### Ruff Linting Rules

Enabled rule sets (see `pyproject.toml`):
- `F` - Pyflakes
- `E`, `W` - PEP 8
- `B` - flake8-bugbear
- `N` - pep8-naming
- `PL` - pylint
- `S` - security (bandit)
- `I` - isort
- `SIM`, `C4`, `ARG`, `ERA`, `RUF`, `G`

Ignored rules:
- `E501` - Line too long (handled by formatter)
- `PLR2004` - Magic values
- `RUF001`, `RUF002`, `RUF003` - Allow Chinese punctuation

Test files ignore: `S101` (assert), `ARG` (unused args)

Use `# noqa: RULE` for justified exceptions with a comment explaining why.

## Git Commit Conventions

- **Language**: All commit messages MUST be in **Chinese**
- **Format**: Conventional commits with prefixes
- **Prefixes**: `[FEATURE]`, `[FIX]`, `[DOCS]`, `[REFACTOR]`, `[TEST]`, `[CHORE]`
- **Example**: `[FEATURE] 添加库伦效率计算功能`

## Architecture Patterns

### Data Flow

```
Raw Files -> IOPluginManager -> RawData + RawDataInfo
                                       |
                                TechniqueAnalyzer.analyze()
                                       |
                               AnalysisData + AnalysisDataInfo
```

### Implementing Analyzers

All analyzers inherit from `TechniqueAnalyzer` and implement `compute()`:

```python
class MyAnalyzer(TechniqueAnalyzer):
    technique = "my_technique"
    required_columns = ("Ewe/V", "<I>/mA")

    def compute(self, raw_data: RawData) -> tuple[AnalysisData, AnalysisDataInfo]:
        ds = raw_data.data
        # ... computation ...
        return AnalysisData(data=result_ds), AnalysisDataInfo(parameters={...})
```

### Adding New Readers

1. Create plugin in `src/echemistpy/io/plugins/[technique]/`
2. Implement `load()` returning `(RawData, RawDataInfo)`
3. Register in `IOPluginManager` with `supported_extensions` and `instrument`
4. Use `# ruff: noqa: N999` for instrument-specific filenames (e.g., `MSPD.py`)

## Key Files

- `src/echemistpy/io/structures.py` - Core data structures (RawData, AnalysisData)
- `src/echemistpy/io/loaders.py` - Unified `load()` interface
- `src/echemistpy/io/plugin_manager.py` - IOPluginManager with pluggy hooks
- `src/echemistpy/processing/analyzers/registry.py` - TechniqueAnalyzer base class
- `pyproject.toml` - Ruff, pytest, and build configuration

## Local Skills Directory

Agent skills are stored at: `C:\Users\chengliu\.claude\skills`

### 安装 Skill 的方法

#### 方法 1: 使用 Skill Manager (推荐)

在对话中直接让 AI 搜索和安装：

```
搜索关于 pytest 的 skills
```

```
安装 pytest-test-generator skill
```

AI 会自动从 31,767+ 社区 skills 中搜索并安装到本地目录。

#### 方法 2: 手动安装

1. **创建 skill 目录**:
   ```powershell
   mkdir C:\Users\chengliu\.claude\skills\<skill-name>
   ```

2. **创建 SKILL.md 文件** (必须全大写):
   ```powershell
   notepad C:\Users\chengliu\.claude\skills\<skill-name>\SKILL.md
   ```

3. **编写 SKILL.md 内容**:
   ```markdown
   ---
   name: my-skill
   description: 简短描述 (1-1024 字符)
   license: MIT
   ---

   # My Skill

   ## 用途
   描述这个 skill 做什么...

   ## 使用场景
   什么时候应该使用这个 skill...
   ```

#### 方法 3: 从 GitHub 下载

使用 SVN (最快):
```powershell
# 安装 SVN: choco install svn
svn export https://github.com/<owner>/<repo>/trunk/<skill-path> C:\Users\chengliu\.claude\skills\<skill-name>
```

使用 Git Sparse Checkout:
```powershell
git clone --filter=blob:none --sparse --depth=1 https://github.com/<owner>/<repo>.git temp-repo
cd temp-repo
git sparse-checkout set <skill-path>
move <skill-path> C:\Users\chengliu\.claude\skills\<skill-name>
```

直接下载 SKILL.md:
```powershell
curl -o C:\Users\chengliu\.claude\skills\<skill-name>\SKILL.md https://raw.githubusercontent.com/<owner>/<repo>/main/<skill-path>/SKILL.md
```

### Skill 命名规则

- 1-64 个字符
- 仅小写字母、数字和单个连字符 `-`
- 不能以 `-` 开头或结尾
- 文件夹名必须与 SKILL.md 中的 `name` 字段一致
- 正则: `^[a-z0-9]+(-[a-z0-9]+)*$`

### 验证安装

重启 OpenCode 后，使用以下命令检查:
```
列出可用的 skills
```

## Copilot Instructions

See `.github/copilot-instructions.md` for additional Chinese-language guidance on:
- Core architecture patterns
- Plugin development workflow
- Data standardization conventions
