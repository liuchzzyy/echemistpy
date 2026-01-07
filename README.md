<h1 align="center">
  <br>
  <img src="docs/static/liuchzzyy.jpg" alt="echemistpy" width="150">
  <br>
  echemistpy
  <br>
</h1>

<p align="center">
<strong>电化学与材料表征的统一数据处理框架</strong><br/>
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

## echemistpy 是什么？

**echemistpy** 是一个为电化学技术与材料表征设计的统一数据处理框架。核心采用可扩展的读取器 + 分析器模式，配合管道编排架构，基于 xarray、numpy 与 scipy 构建。

### 主要特性

- **统一数据格式**：以 `xarray.Dataset` (扁平数据) 和 `xarray.DataTree` (层级数据) 表示所有实验数据
- **读取器接口**：面向不同仪器/格式的可扩展数据读取与标准化
- **模块化分析器**：模板化实现，便于扩展新技术分析
- **管道编排**：批量处理多测量样品，自动汇总结果
- **类型安全配置**：traitlets 保障配置校验与一致性
- **插件架构**：pluggy 注册机制，灵活扩展技术支持

> **⚠️ 注意**：echemistpy 正在积极开发中。设计可能会进行更改。请在 [Issue Tracker](https://github.com/liuchzzyy/echemistpy/issues) 中报告问题。

---

## 快速开始

### 安装（推荐使用 uv）

项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理和虚拟环境构建。

```powershell
# 1. 同步依赖（自动创建/更新虚拟环境）
uv sync

# 2. 安装所有可选依赖分组（文档、交互环境等）
uv sync --all-groups

# 3. 激活环境
.venv\Scripts\activate
```

### 开发流程

提交代码前，请确保通过以下质量检查：

```powershell
# 格式化代码
uv run ruff format src/

# 代码质量检查（Lint）
uv run ruff check src/ --fix

# 静态类型检查
uv run ty check

# 运行测试
uv run pytest
```

---

## 使用示例

### 数据读取与标准化 (I/O)

`echemistpy` 提供了一个统一的 `load` 接口，可以自动识别文件格式并将其标准化为统一的列名和单位。

```python
from echemistpy.io import load

# 加载 BioLogic .mpt 文件
raw_data, raw_info = load("docs/examples/echem/Biologic_GPCL.mpt", sample_name="MySample")

# 查看标准化后的数据 (xarray.Dataset)
print(raw_data.data)

# 查看元数据
print(raw_info.to_dict())
```

### 分析数据

```python
from echemistpy.processing.analyzers.echem import GalvanostaticAnalyzer

# 创建分析器
analyzer = GalvanostaticAnalyzer()

# 运行分析
result_data, result_info = analyzer.analyze(raw_data)
```

---

## 核心架构

```
Raw Files -> IOPluginManager -> RawData + RawDataInfo
                                       |
                                TechniqueAnalyzer.analyze()
                                       |
                               AnalysisData + AnalysisDataInfo
```

- **IOPluginManager**: 自动识别文件格式并分发给对应的 `Reader`
- **RawData**: 存储原始测量数据，基于 `xarray`
- **TechniqueAnalyzer**: 分析算法基类，处理数据验证、预处理和计算
- **AnalysisData**: 存储分析结果，保持接口统一

---

## 项目结构

```
echemistpy/
├── src/echemistpy/
│   ├── io/                  # 核心 I/O 系统
│   │   ├── plugins/         # 仪器/格式读取插件
│   │   ├── loaders.py       # 统一 load() 接口
│   │   ├── structures.py    # 数据结构 (RawData, AnalysisData)
│   │   └── plugin_manager.py # 插件管理器
│   ├── processing/          # 数据处理
│   │   ├── analyzers/       # 各种技术的分析算法
│   │   └── pipeline.py      # 分析管道编排
│   └── utils/               # 工具函数
├── tests/                   # 单元测试
├── docs/                    # 文档和示例 Notebooks
└── pyproject.toml           # 项目与依赖配置
```

---

## 贡献指南

欢迎提交问题和拉取请求！

### 开发流程

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/my-feature`
3. 提交更改（**中文 Commit**，格式：`[FEATURE]`, `[FIX]`, `[DOCS]` 等）
4. 运行完整检查：格式化、Lint、类型检查、测试
5. 提交拉取请求

详细开发指南请参考 [AGENTS.md](AGENTS.md)。

---

## 许可证

[CeCILL-B 免费软件许可协议](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)

更多详情请参见仓库中的 [LICENSE](LICENSE) 文件。

---

## 引用

```bibtex
@software{echemistpy,
  author = {Cheng Liu},
  title = {echemistpy},
  url = {https://github.com/liuchzzyy/echemistpy},
  year = {2025}
}
```

**最后更新**: 2026 年 1 月 7 日
