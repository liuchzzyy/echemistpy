# Copilot Instructions for echemistpy

## 项目概述

echemistpy 是一个科学数据分析库，专注于电化学技术和材料表征（XRD、XPS、TGA、XAS, TXM）等实验技术的统一数据处理。核心架构采用可扩展的分析器模式 + 管道编排设计，基于 xarray、numpy 和 scipy 构建。

### 核心技术栈

- **xarray**: 多维标注数组，支持 `Dataset`（扁平数据）和 `DataTree`（分层数据，如 XRD）
- **numpy** 和 **scipy**: 数值计算和数组操作
- **traitlets**: 类型验证和配置管理，用于元数据管理
- **pandas**: 用于表格数据处理和时间序列转换
- **pluggy**: 插件系统，用于扩展注册机制

## 核心架构模式

### 数据流设计

```
原始数据文件 → IOPluginManager → RawData + RawDataInfo
                                        ↓
                                 TechniqueAnalyzer.analyze()
                                        ↓
                                AnalysisData + AnalysisDataInfo
```

**核心数据结构** (`io/structures.py`):

- **RawData** + **RawDataInfo**: 测量数据的容器，包含 `xarray.Dataset` 或 `xarray.DataTree` 以及元数据
- **AnalysisData** + **AnalysisDataInfo**: **统一的**分析结果容器（适用于所有分析类型）
- 后端：`xarray.Dataset` 用于扁平数据，`xarray.DataTree` 用于分层数据（如不同温度下的 XRD 扫描）
- 元数据：使用 traitlets 进行验证和动态参数存储
- 所有容器继承自 `BaseData`/`BaseInfo`，共享混入类（`XarrayDataMixin`, `MetadataInfoMixin`）

**I/O 系统** (`io/`):

- **IOPluginManager**: 基于插件的系统，支持不同仪器
- **load()**: 统一加载接口，自动格式检测
- **DataStandardizer**: 规范化列名、单位和时间格式
- 支持文件和目录作为输入

**处理系统** (`processing/`):

- **TechniqueAnalyzer**: 所有分析器的抽象基类
  - `analyze()`: 模板方法，编排验证 → 预处理 → 计算 → 元数据继承
  - `compute()`: **必须实现**的抽象方法，返回 `tuple[AnalysisData, AnalysisDataInfo]`
  - `validate()`: 可选的数据验证（检查 `required_columns`）
  - `preprocess()`: 可选的数据预处理
- **TechniqueRegistry**: 将技术和仪器标识符映射到分析器实例
- **AnalysisPipeline**: 高级编排，带日志和错误处理

### 统一的分析器接口

**所有分析器遵循单一、一致的模式**：

**基类职责** (`TechniqueAnalyzer.analyze()`):

1. 验证输入数据
2. 预处理数据（在副本上）
3. 调用 `compute()` 获取 `AnalysisData + AnalysisDataInfo`
4. 从 `raw_info` 继承元数据（sample_name、start_time、operator 等）
5. 更新 `technique` 字段
6. 返回 `AnalysisData + AnalysisDataInfo`

**子类职责**:

- 实现 `compute()` 返回 `tuple[AnalysisData, AnalysisDataInfo]`
- 可选择覆盖 `validate()` 和 `preprocess()`
- 设置 `technique` 类属性和 `required_columns` 元组

### 分析器实现模式

```python
from echemistpy.processing.analyzers.registry import TechniqueAnalyzer
from echemistpy.io.structures import AnalysisData, AnalysisDataInfo, RawData

class MyAnalyzer(TechniqueAnalyzer):
    technique = "my_technique"  # 在 TechniqueRegistry 中注册
    required_columns = ("Ewe/V", "<I>/mA")  # 验证输入数据

    def compute(self, raw_data: RawData) -> tuple[AnalysisData, AnalysisDataInfo]:
        # 提取和处理数据
        ds = raw_data.data

        # 执行计算
        result_ds = ds.assign({
            "processed": ds["Ewe/V"] * ds["<I>/mA"]
        })

        # 打包结果
        analysis_data = AnalysisData(data=result_ds)
        analysis_info = AnalysisDataInfo(parameters={
            "scaling_factor": 1.0,
            "method": "simple_product"
        })

        return analysis_data, analysis_info

    def validate(self, raw_data: RawData) -> None:
        # 可选：添加自定义验证
        super().validate(raw_data)

    def preprocess(self, raw_data: RawData) -> RawData:
        # 可选：添加自定义预处理，返回修改后的副本
        return raw_data
```

**关键点**:

- 基类 `analyze()` 自动处理元数据继承
- 您**只需**实现 `compute()` 返回 `AnalysisData + AnalysisDataInfo`
- 无需手动复制 `sample_name`、`start_time` 等 - 基类会处理
- 适用于所有分析类型：电化学、XRD、XPS 等

### 数据加载流程

1. **统一入口**: 使用 `load(path, instrument=None, **kwargs)` 加载数据。
2. **插件检索**: `IOPluginManager` 根据文件后缀匹配插件。如果后缀对应多个仪器（如 `.xlsx`），必须显式指定 `instrument`。
3. **标准化**: `load()` 默认调用 `DataStandardizer` 统一列名、单位和时间格式。

## 开发工作流

### 环境配置

项目使用 **uv** 进行依赖管理和环境配置：

```powershell
# 安装核心依赖并创建虚拟环境
uv sync

# 安装开发工具（ruff、pytest、pre-commit）
uv sync --only-group dev

# 安装文档工具
uv sync --only-group docs

# 安装 Jupyter/交互式工具
uv sync --only-group interactive

# 安装所有依赖组
uv sync --all-groups
```

核心依赖包括 `xarray`、`traitlets`、`pandas`、`numpy`、`scipy`。

### 代码质量与类型检查

项目强制执行严格的代码质量检查：

- **Ruff**: 用于静态分析和格式化
  - 运行: `uv run ruff check src/` 和 `uv run ruff format src/`
  - 配置: 行长度 200，扩展检查（Pyflakes, PEP 8, flake8-bugbear, pep8-naming, pylint, security）
  - 测试文件忽略: `S101`（assert 使用）、`ARG`（未使用参数）
  - 配置文件: `pyproject.toml`
- **Ty**: 用于静态类型检查
  - 运行: `uv run ty check`
  - 重点: 确保 `RawData` 的 `data` 属性在处理 `Dataset` 和 `DataTree` 时类型正确
- **测试**:
  - 运行所有测试: `uv run pytest`
  - 单个测试: `uv run pytest tests/integration/test_io_with_real_data.py::test_biologic_gpcl`
  - 覆盖率报告: `uv run pytest --cov=echemistpy`

### Git 提交规范

- **语言**: 所有 Git commit 消息必须使用 **中文**
- **格式**: 遵循约定式提交 (Conventional Commits) 规范，使用前缀如 `[FEATURE]`、`[FIX]`、`[DOCS]` 等
- **示例**: `[FEATURE] 添加库伦效率计算功能`

## 扩展点和集成

### 支持的仪器和格式

**电化学**

- BioLogic (.mpt) - `BiologicMPTReader`
- LANHE (.xlsx) - `LanheXLSXReader`

**材料表征**

- XRD: MSPD (.xye) - `MSPDReader`
- XAS: CLAESS (.dat) - `CLAESSReader`
- TXM: MISTRAL (.hdf5) - `MistralHDF5Reader`

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

## 重要说明

- **所有分析器**返回 `AnalysisData + AnalysisDataInfo` - 这是统一的接口
- 基类 `analyze()` 方法自动处理从 `raw_info` 的元数据继承
- 使用 traitlets 进行运行时验证和类型安全
- 插件系统使用 pluggy 实现灵活的扩展注册
- 公共 API 使用英文；内部代码使用中文注释
- 项目状态: **Alpha** - 设计可能在不同版本间变化
