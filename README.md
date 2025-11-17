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
	pandas>=2.3.1 `
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
