### 快速开始

以下步骤可在 Windows PowerShell 中完成，约几分钟即可准备好开发环境。

1. **安装 Miniconda**
   ```powershell
   winget install -e --id Anaconda.Miniconda3
   ```
2. **接受默认频道 ToS（仅需一次）**
   ```powershell
   & "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   & "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   & "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
   ```
3. **创建并配置环境**
   ```powershell
   & "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" create -y -n txm python=3.10
   & "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" config --add channels spectrocat
   & "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" config --add channels conda-forge
   & "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" install -n txm -y
        exspy>=0.3.2
        h5netcdf>=1.6.3
        "hyperspy[all]>=2.3.0"
        ipykernel>=6.30.0
        ipympl>=0.9.7
        jupyterlab>=4.4.5
        matplotlib>=3.10.3
        numpy>=2.2.6
        openpyxl>=3.1.5
        ruff>=0.12.4
        scipy>=1.15.3
        sparse>=0.17.0
        spectrochempy==0.7.1
        uncertainties>=3.2.3
        xarray>=2025.6.1
   conda activate txm
   ```

环境默认位于 `C:\Users\<you>\AppData\Local\miniconda3\envs\txm`。

### echemistpy 包概览

`echemistpy`（位于 `src/`）提供统一的数据模型和分析管线，支持 XRD、XPS、TGA、CV 等技术。

#### 核心 IO 工作流 (Read -> Standardize -> Save)

`echemistpy` 采用三步数据处理流程：
1. **Read**: 使用 `load_data_file` 读取原始文件，生成包含原始数据和元数据的 `RawMeasurement`。
2. **Standardize**: 使用 `standardize_measurement` 将原始数据转换为标准化的 `Measurement`（数据）和 `MeasurementInfo`（元数据）。
3. **Save**: 使用 `save_measurement` 将标准化数据保存为 CSV（带元数据头）或 HDF5/NetCDF（完整数据结构）。

典型工作流程：

```python
from pathlib import Path
from echemistpy.io import load_data_file, standardize_measurement, save_measurement

# 1. 读取数据 (自动检测格式)
raw_meas = load_data_file("raw_data.json")

# 2. 标准化 (自动识别技术并重命名列，例如 Time -> Time/s)
meas, info = standardize_measurement(raw_meas)

print(f"Technique: {info.technique}")
print(f"Standardized Vars: {list(meas.data.data_vars)}")

# 3. 保存数据
# 保存为 CSV (适用于 1D/2D 表格数据，元数据作为注释头)
save_measurement(meas, info, "output.csv")

# 保存为 NetCDF (完整保存所有数据和元数据)
save_measurement(meas, info, "output.nc")
```

- **数据结构**:
    - `RawMeasurement`: 原始数据容器。
    - `Measurement`: 标准化后的数据 (`xarray.Dataset`)。
    - `MeasurementInfo`: 标准化后的元数据 (`dataclass`)。
    - `Results` / `ResultsInfo`: 分析结果及其元数据。
- **扩展性**: `utils/` 下包含 `math/`、`plotting/`、`external/`，用于数值算法与可视化扩展。

### 仓库结构

当前仓库采用标准的 Python 包布局，以便于发布、测试与文档维护：

```
echemistpy/
├── src/echemistpy/        # 核心源码
├── tests/                 # Pytest 测试（含占位示例）
├── docs/                  # 文档与 Characterization 资料
├── examples/              # 最小可运行的使用示例
├── LICENSE                # 专有授权声明
├── CHANGELOG.md           # 版本更新记录
├── CONTRIBUTING.md        # 贡献指南
└── MANIFEST.in            # 打包额外文件清单
```

`pyproject.toml`（PEP 621）声明了构建系统与依赖，`.gitignore` 覆盖了常见的 Python 工件，方便直接接入 CI/CD。
