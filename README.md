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

`echemistpy`（位于 `src/`）提供统一的数据模型和分析管线，支持 XRD、XPS、TGA、CV 等技术。典型工作流程：

```python
from pathlib import Path

import numpy as np
import xarray as xr

from echemistpy import (
    AnalysisPipeline,
    Measurement,
    MeasurementMetadata,
    default_registry,
)

theta = np.linspace(10, 80, 200)
intensity = np.sin(theta / 10) ** 2

data = xr.Dataset({
    "2theta": ("row", theta),
    "intensity": ("row", intensity),
})
metadata = MeasurementMetadata(technique="xrd", sample_name="Sample-01")
measurement = Measurement(data=data, metadata=metadata)

pipeline = AnalysisPipeline(default_registry)
results = pipeline.run([measurement])
print(pipeline.summary_table(results))
```

- 新增技术：继承 `TechniqueAnalyzer` 并注册到 `TechniqueRegistry`。
- `utils/` 下包含 `math/`、`plotting/`、`external/`，用于数值算法与可视化扩展。
- `io.load_table`/`io.save_table` 可在 `xarray.Dataset` 与 `csv/NetCDF` 等格式间转换。

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
