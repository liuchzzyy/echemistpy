# Copilot Instructions for echemistpy

## 项目概述

echemistpy 是一个科学数据分析库，专注于电化学技术和材料表征（XRD、XPS、TGA、XAS, TXM）等实验技术的统一数据处理。核心架构采用可扩展的分析器模式 + 管道编排设计。

### 核心技术栈

- **xarray**: 多维标注数组，支持 `Dataset` 和 `DataTree` (用于分层数据如 XRD)。
- **numpy**: 数值计算和数组操作。
- **traitlets**: 类型验证和配置管理，用于插件系统和元数据。
- **pandas**: 用于表格数据处理和时间序列转换。

## 核心架构模式

### 数据流设计

- **RawData** + **RawDataInfo** (`io/structures.py`): 测量数据的容器，包含 `xarray.Dataset` 或 `xarray.DataTree` 以及元数据。
- **ResultsData** + **ResultsDataInfo** (`io/structures.py`): 分析后的结果容器。
- **TechniqueAnalyzer** (`processing/analyzers/base.py`): 所有分析器的抽象基类。
- **AnalysisPipeline**: 协调加载、分析和聚合的高级编排器。
- **IOPluginManager** (`io/plugin_manager.py`): 管理所有数据读取插件，支持按文件扩展名和仪器名称 (instrument) 检索。

### 分析器实现模式

继承 `TechniqueAnalyzer` 时必须实现 `compute` 方法，返回摘要字典和结果表格字典。

### 数据加载流程

1. **统一入口**: 使用 `load(path, instrument=None, **kwargs)` 加载数据。
2. **插件检索**: `IOPluginManager` 根据文件后缀匹配插件。如果后缀对应多个仪器（如 `.xlsx`），必须显式指定 `instrument`。
3. **标准化**: `load()` 默认调用 `DataStandardizer` 统一列名、单位和时间格式。

## 开发工作流

### 环境配置

使用 Conda 环境 "txm"。核心依赖包括 `xarray`, `traitlets`, `pandas`, `numpy`。

### 代码质量与类型检查

项目强制执行严格的代码质量检查：

- **Ruff**: 用于静态分析和格式化。
  - 运行: `uv run ruff check` 和 `uv run ruff format`。
  - 配置文件: `pyproject.toml`。
- **Ty**: 用于静态类型检查。
  - 运行: `uv run ty check`。
  - 重点: 确保 `RawData` 的 `data` 属性在处理 `Dataset` 和 `DataTree` 时类型正确。

### Git 提交规范

- **语言**: 所有 Git commit 消息必须使用 **中文**。
- **格式**: 遵循约定式提交 (Conventional Commits) 规范。

## 扩展点和集成

### 添加新读取器 (IO Plugin)

1. 在 `src/echemistpy/io/plugins/` 下按技术分类创建插件（如 `echem/`, `xrd/`）。
2. 实现读取逻辑，返回 `RawData` 和 `RawDataInfo`。
3. 在 `src/echemistpy/io/plugin_manager.py` 中注册插件，指定 `supported_extensions` 和 `instrument`。
4. **命名约定**: 遵循 PEP 8，但特定仪器类文件（如 `MSPD.py`）可保持原样并添加 `# ruff: noqa: N999`。

### 数据标准化约定

所有读取器输出应通过 `DataStandardizer` 进行标准化：

- **时间列**:
  - `systime`: 绝对时间 (datetime64)。
  - `time_s`: 相对时间（秒，float64）。
- **电化学列**:
  - `Ewe/V`: 电位。
  - `<I>/mA`: 电流。
- **坐标名称**: 统一使用 `"record"` 或 `"row"` 作为主要维度。

## 项目特定约定

### 文件组织

- 插件位于 `src/echemistpy/io/plugins/`。
- 分析器位于 `src/echemistpy/processing/analyzers/`。
- 核心数据结构位于 `src/echemistpy/io/structures.py`。

### 常见模式

#### 使用 load() 加载数据

```python
from echemistpy.io import load

# 自动检测
raw = load("data.mpt")

# 显式指定仪器（处理歧义格式如 .xlsx）
raw = load("data.xlsx", instrument="lanhe")
```

#### 处理 DataTree (XRD)

```python
# XRD 数据通常存储在 DataTree 中以支持多层级（如不同温度下的扫描）
if raw.is_tree:
    tree = raw.data  # xr.DataTree
    # 访问特定节点
    node = tree["scan_1"]
```
