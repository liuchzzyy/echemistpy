# Copilot Instructions for echemistpy

## 项目概述

echemistpy 是一个科学数据分析库，专注于电化学技术和材料表征（XRD、XPS、TGA、XAS, TXM）等实验技术的统一数据处理。核心架构采用可扩展的分析器模式 + 管道编排设计。

### 核心技术栈

- **xarray**: 多维标注数组，所有数据的内部表示
- **numpy**: 数值计算和数组操作
- **pluggy**: 插件系统，支持可扩展的读取器和分析器架构
- **traitlets**: 类型验证和配置管理

## 核心架构模式

### 数据流设计

- **RawData** + **RawDataInfo** (`io/structures.py`): 测量数据的容器，包含 `xarray.Dataset` 和元数据。`load()` 函数会自动读取并标准化这些数据。
- **ResultsData** + **ResultsDataInfo** (`io/structures.py`): 分析后的结果，包含 `xarray.Dataset` 和实验元数据。
- **TechniqueAnalyzer** (`processing/analysis/base.py`): 所有分析器的抽象基类，实现 `analyze()` 方法。
- **AnalysisPipeline** (`pipelines/manager.py`): 协调加载、分析和聚合的高级编排器。
- **TechniqueRegistry** (`processing/analysis/registry.py`): 技术标识符到分析器实例的映射注册表。

### 分析器实现模式

继承 `TechniqueAnalyzer` 时必须实现：

```python
class CustomAnalyzer(TechniqueAnalyzer):
    technique = "custom_tech"  # 技术标识符

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("col1", "col2")  # 必需的数据列

    def compute(self, raw_data: RawData) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        # 返回 (summary_dict, tables_dict)
        return summary, tables
```

查看 [analysis/echem.py](src/echemistpy/processing/analysis/echem.py) 的 `CyclicVoltammetryAnalyzer` 作为完整示例。

### 数据加载流程

1. **加载与标准化**: 使用 `load()` 自动处理整个流程（读取原始数据 -> 自动检测技术 -> 标准化列名和元数据）。
2. **结果**: 返回 `RawData` + `RawDataInfo`。

如果需要完全未处理的数据，可以使用 `load(path, standardize=False)`。

## 开发工作流

### 环境配置

使用 Conda 环境 "txm" (见 `environment.yml` 和 `pyproject.toml`)：

```powershell
# 快速安装 (参见 README.md 完整版本)
winget install -e --id Anaconda.Miniconda3
& "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" create -y -n txm python=3.10
& "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" config --add channels spectrocat --add channels conda-forge
& "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" install -n txm -y exspy hyperspy[all] jupyterlab matplotlib numpy scipy xarray galvani
conda activate txm
```

注意：

- 环境位置：`C:\Users\<you>\AppData\Local\miniconda3\envs\txm`
- 核心依赖：`xarray`, `galvani` (BioLogic), `hyperspy`, `spectrochempy`
- 开发工具：`ruff` 用于代码质量，`pytest` 用于测试

### 测试策略

- 现有测试: [test_biologic_reader.py](tests/test_biologic_reader.py)、[test_lanhe_reader.py](tests/test_lanhe_reader.py)
- 使用 [examples/echem/](examples/echem/) 中的真实仪器数据文件进行集成测试
- 运行测试: `python -m pytest tests/`
- 预期测试: doctest 验证、综合读取器测试 (将来添加)

### 代码质量

使用 Ruff (配置在 `pyproject.toml`)：

```powershell
ruff check src/ tests/  # 静态分析
ruff format src/ tests/  # 代码格式化
```

## 扩展点和集成

### 添加新技术支持

1. 在 [processing/analysis/](src/echemistpy/processing/analysis/) 下创建新的分析器模块
2. 继承 `TechniqueAnalyzer` 并实现必需方法
3. 在 [analysis/registry.py](src/echemistpy/processing/analysis/registry.py) 的 `create_default_registry()` 中注册

### 外部数据读取器

位于 [utils/external/echem/](src/echemistpy/utils/external/echem/)，例如：

- `biologic_reader.py`: BioLogic .mpt/.mpr 文件解析器 (`BiologicMPTReader` 类)
- `lanhe_reader.py`: LANHE .ccs 文件解析器 (`LanheReader` 类)

新读取器应返回 `RawData` + `RawDataInfo` 对象，包含正确的元数据和轴定义。读取器使用状态机模式 (`_ReaderState`) 处理复杂文件格式。

### I/O 模式

- [io/loaders.py](src/echemistpy/io/loaders.py): `load()` 协调整个数据加载和标准化流程
- [io/saver.py](src/echemistpy/io/saver.py): `save_table()` 导出 `xarray.Dataset` 到各种格式
- 所有数据使用 `xarray.Dataset` 作为内部表示，坐标名为 "row"
- 数据标准化: `standardize_names()` 统一不同仪器的列名和单位

## 项目特定约定

### 文件组织

- Jupyter notebooks 在 [docs/Characterization/](docs/Characterization/) 按技术分类
- 示例数据文件在 [examples/echem/](examples/echem/)
- 所有源码使用绝对导入: `from echemistpy.io import AnalysisData`
- 分析器模块在 [processing/analysis/](src/echemistpy/processing/analysis/) 下

### 数据约定

- 时间列统一使用 `"time/s"` 标识符 (`t_str` 常量)
- 电压列使用 `"Ewe/V"` (工作电极电位) 或 `"potential"`
- 电流列使用 `"<I>/mA"` 或 `"current"`
- 所有数据维度名称为 `"row"`
- 数据预处理：基线校正 (`baseline_corrected`)、归一化 (`normalized_current`)
- 分析器输出：`summary` (字典) + `tables` (xarray.Dataset 字典)
- 读取器输出：`RawData` (包含 `xarray.Dataset`) + `RawDataInfo` (包含元数据字典)

### 测试数据

使用 [examples/echem/](examples/echem/) 中的真实仪器文件进行集成测试，确保读取器能处理实际实验室数据格式。

## 常见模式

### 创建 AnalysisData 对象

```python
metadata = AnalysisDataInfo(technique="xrd", sample_name="Sample-01")
analysis_data = AnalysisData(data=xr_dataset)
```

### 运行分析管道

```python
pipeline = AnalysisPipeline(default_registry)
results = pipeline.run([summary])
summary_table = pipeline.summary_table(results)
```

### 处理分析结果

`ResultsData` 包含 `summary` (字典)、`tables` (xarray.Dataset 字典) 和可选的 `figures`。

### 读取仪器数据文件

```python
from echemistpy.utils.external.echem.biologic_reader import BiologicMPTReader

reader = BiologicMPTReader("path/to/file.mpt")
raw_data, raw_data_info = reader.read()  # 返回 RawData 和 RawDataInfo
```
