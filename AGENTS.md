# AGENTS.md

## 核心命令 (使用 `uv`)

```powershell
uv sync --all-groups        # 环境同步
uv run ruff format src/     # 格式化
uv run ruff check src/ --fix # Lint 修复
uv run ty check             # 类型检查 (严格)
uv run pytest               # 运行测试
```

## 强制规范

### 1. 代码质量流程
提交前必须依次执行：`format` -> `check` -> `ty check` -> `test`。

### 2. 类型与风格
- **类型注解**：所有函数必须包含类型注解。优先使用 modern syntax (`list[str]`, `str | None`)。
- **命名**：类使用 `PascalCase`，函数/变量使用 `snake_case`。
- **导包顺序**：Future -> Standard -> Third-party -> Local (空行分隔)。
- **Docstrings**: 使用 Google Style。公有 API 用英文，内部实现可用中文。

### 3. Git 提交
- **语言**：必须使用**中文**。
- **格式**：`[PREFIX] 描述`。
- **前缀**：`[FEATURE]`, `[FIX]`, `[DOCS]`, `[REFACTOR]`, `[TEST]`, `[CHORE]`。
- **示例**:
    - `[FEATURE] 添加 Biologic MPT 文件读取支持`
    - `[FIX] 修复 XAS 数据对齐时的维度错误`


## 架构核心

### 数据容器 (`io/structures.py`)
- **数据存储**: 使用 `RawData` / `AnalysisData`。底层是 `xarray.Dataset` (扁平) 或 `xarray.DataTree` (层级)。
- **元数据**: 使用 `RawDataInfo` / `AnalysisDataInfo`。基于 `traitlets.HasTraits`，支持标准字段和动态 `parameters`/`others`。

### I/O 扩展 (`io/plugins/`)
- **Reader**: 必须继承 `BaseReader` 或遵循其接口。
- **注册**: 在 `plugin_manager.py` 中通过 `pm.register_loader(extensions, class)` 注册。
- **标准化**: 确保列名和单位在读取时通过 `column_mappings.py` 转换。

### 分析器扩展 (`processing/analyzers/`)
- **基类**: 必须继承 `TechniqueAnalyzer`。
- **实现**: 
    - `required_columns`: 定义必需的数据列。
    - `_compute(self, raw_data)`: 实现核心算法。
- **调用**: 统一通过 `analyzer.analyze(raw_data, raw_info)` 调用以确保元数据继承。

### 新增模块 (XAS 支持)
- **预处理 (`processing/preprocessing/xas.py`)**:
    - `calibrate_energy`: 能量校准
    - `align_spectra`: 谱图对齐
    - `deglitch`, `smooth`: 数据清洗
    - `correct_fluorescence`: 自吸收校正
- **数学工具 (`processing/math/fitting.py`)**:
    - PCA, NMF (降维)
    - LCF (线性组合拟合)
- **分析器 (`processing/analyzers/xas.py`)**:
    - `XASAnalyzer`: 封装 `xraylarch` (Normalization, AutoBK, FFT)
- **可视化 (`visualization/plot_xas.py`)**:
    - `plot_echem_xas`: 电化学-光谱同步图 (LC Plot)

## 推荐库
- 拟合：`pybaselines`, `lmfit`
- 图像：`scikit-image`
- 机器学习：`scikit-learn`, `umap-learn`
- XAS 分析：`xraylarch`
