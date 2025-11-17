# Copilot Instructions for echemistpy

## 项目概述

echemistpy 是一个科学数据分析库，专注于电化学（XRD、XPS、TGA、CV）等实验技术的统一数据处理。核心架构采用可扩展的分析器模式 + 管道编排设计。

## 核心架构模式

### 数据流设计
- **Measurement** (`io/structures.py`): 包含 `xarray.Dataset`、`MeasurementMetadata` 和 `Axis` 列表的容器
- **TechniqueAnalyzer** (`analysis/base.py`): 所有分析器的抽象基类，实现 `analyze()` 方法
- **AnalysisPipeline** (`pipelines/manager.py`): 协调加载、分析和聚合的高级编排器
- **TechniqueRegistry** (`analysis/registry.py`): 技术标识符到分析器实例的映射注册表

### 分析器实现模式
继承 `TechniqueAnalyzer` 时必须实现：
```python
class CustomAnalyzer(TechniqueAnalyzer):
    technique = "custom_tech"  # 技术标识符
    
    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("col1", "col2")  # 必需的数据列
    
    def compute(self, measurement: Measurement) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        # 返回 (summary_dict, tables_dict)
```

查看 `analysis/echem.py` 的 `CyclicVoltammetryAnalyzer` 作为完整示例。

## 开发工作流

### 环境配置
使用 Conda 环境 "txm" (见 `environment.yml`)：
```powershell
conda create -n txm python=3.10
conda config --add channels spectrocat --add channels conda-forge
conda install -n txm -y exspy hyperspy[all] jupyterlab matplotlib numpy scipy xarray
```

### 测试策略
- `tests/test_doctests.py`: 验证所有公开模块的 docstring 示例
- `tests/test_echem_readers_comprehensive.py`: 使用 `examples/echem/` 中真实数据文件的综合测试
- 运行测试: `python -m pytest tests/`

### 代码质量
使用 Ruff (配置在 `pyproject.toml`)：
```powershell
ruff check src/ tests/  # 静态分析
ruff format src/ tests/  # 代码格式化
```

## 扩展点和集成

### 添加新技术支持
1. 在 `analysis/` 下创建新的分析器模块
2. 继承 `TechniqueAnalyzer` 并实现必需方法
3. 在 `analysis/registry.py` 的 `create_default_registry()` 中注册

### 外部数据读取器
位于 `utils/external/echem/`，例如：
- `biologic_reader.py`: BioLogic .mpt/.mpr 文件解析器
- `lanhe_reader.py`: LANHE .ccs 文件解析器

新读取器应返回 `Measurement` 对象，包含正确的元数据和轴定义。

### I/O 模式
- `io/loaders.py`: `load_table()` 支持 CSV/TSV/JSON/NetCDF 格式
- `io/saver.py`: `save_table()` 导出 `xarray.Dataset` 到各种格式
- 所有数据使用 `xarray.Dataset` 作为内部表示，坐标名为 "row"

## 项目特定约定

### 文件组织
- Jupyter notebooks 在 `docs/Characterization/` 按技术分类
- 示例数据文件在 `examples/echem/` 
- 所有源码使用绝对导入: `from echemistpy.io import Measurement`

### 数据约定
- 时间列统一使用 `"time/s"` 标识符
- 电压列使用 `"Ewe/V"` (工作电极电位)
- 电流列使用 `"<I>/mA"` 或 `"current"`
- 所有数据维度名称为 `"row"`

### 测试数据
使用 `examples/echem/` 中的真实仪器文件进行集成测试，确保读取器能处理实际实验室数据格式。

## 常见模式

### 创建 Measurement 对象
```python
metadata = MeasurementMetadata(technique="xrd", sample_name="Sample-01")
measurement = Measurement(data=xr_dataset, metadata=metadata)
```

### 运行分析管道
```python
pipeline = AnalysisPipeline(default_registry)
results = pipeline.run([measurement])
summary_table = pipeline.summary_table(results)
```

### 处理分析结果
`AnalysisResult` 包含 `summary` (字典)、`tables` (xarray.Dataset 字典) 和可选的 `figures`。
