## 环境配置说明

最后更新：2025-10-11

以下步骤基于 Windows PowerShell。

### 1. 安装 Miniconda

```powershell
winget install -e --id Anaconda.Miniconda3
```

### 2. 接受 Anaconda 默认频道的 ToS

```powershell
& "C:\Users\chengliu\AppData\Local\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
& "C:\Users\chengliu\AppData\Local\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
& "C:\Users\chengliu\AppData\Local\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

### 3. 创建 `txm` 环境

```powershell
& "C:\Users\chengliu\AppData\Local\miniconda3\Scripts\conda.exe" create -y -n txm python=3.10
```

### 4. 添加额外频道

```powershell
& "C:\Users\chengliu\AppData\Local\miniconda3\Scripts\conda.exe" config --add channels spectrocat
& "C:\Users\chengliu\AppData\Local\miniconda3\Scripts\conda.exe" config --add channels conda-forge
```

### 5. 安装项目依赖

```powershell
& "C:\Users\chengliu\AppData\Local\miniconda3\Scripts\conda.exe" install -n txm -y `
	exspy>=0.3.2 `
	h5netcdf>=1.6.3 `
	h5py>=3.14.0 `
	"hyperspy[all]>=2.3.0" `
	ipykernel>=6.30.0 `
	ipympl>=0.9.7 `
	jupyterlab>=4.4.5 `
	matplotlib>=3.10.3 `
	numpy>=2.2.6 `
        openpyxl>=3.1.5 `
	ruff>=0.12.4 `
	scipy>=1.15.3 `
	seaborn>=0.13.2 `
	sparse>=0.17.0 `
	spectrochempy==0.7.1 `
	uncertainties>=3.2.3 `
	xarray>=2025.6.1
```

### 6. 启用环境

```powershell
conda activate txm
```

环境路径：`C:\Users\chengliu\AppData\Local\miniconda3\envs\txm`。

## echemistpy 包简介

该仓库现在包含一个可安装的 `echemistpy` Python 包（位于 `src/` 目录）。
它提供了一套可拓展的数据模型、技术分析器（XRD、XPS、TGA、CV 等）和
分析管线，便于在 Jupyter Notebook 或脚本中重用。典型的使用方式：

```python
from pathlib import Path

import numpy as np
import xarray as xr

from echemistpy import AnalysisPipeline, Measurement, MeasurementMetadata, default_registry

# 构造一个简单的 xarray 数据集（真实场景可以通过 echemistpy.io.load_table 加载）
theta = np.linspace(10, 80, 200)
intensity = np.sin(theta / 10) ** 2
data = xr.Dataset(
    {
        "2theta": ("row", theta),
        "intensity": ("row", intensity),
    }
)
metadata = MeasurementMetadata(technique="xrd", sample_name="Sample-01")
 measurement = Measurement(data=data, metadata=metadata)

 pipeline = AnalysisPipeline(default_registry)
 results = pipeline.run([measurement])
 summary = pipeline.summary_table(results)
 print(summary)
```

如需添加新的技术，只需继承 `TechniqueAnalyzer` 并将其实例注册到
`TechniqueRegistry` 即可。`utils/` 目录下已经细分为 `math/`、`plotting/`
和 `external/` 子模块，方便未来扩展数值算法、可视化和外部服务。`io`
模块提供 `load_table` 与 `save_table`，可直接在 `xarray.Dataset` 与多种
存储格式（CSV/TSV/JSON/NetCDF）之间往返转换，确保分析管线自始至终
都使用统一的数据格式。新增的电化学（Cyclic Voltammetry）分析器能
自动给出氧化/还原峰和净电荷等关键指标，便于快速筛选样品。
