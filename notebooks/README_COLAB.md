# 在 Google Colab 中运行 echemistpy

本指南说明如何在 Google Colab 中运行 echemistpy 的 Jupyter Notebooks。

## 🚀 一键启动

### STXMAnalyzer 示例

点击下面的按钮在 Colab 中打开：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liuchzzyy/echemistpy/blob/cl_version/notebooks/test_stxm_analyzer.ipynb)

### 其他示例

- **电化学分析**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liuchzzyy/echemistpy/blob/cl_version/notebooks/test_echem_analyzer.ipynb)

## 📋 手动打开步骤

1. 访问 [Google Colab](https://colab.research.google.com/)
2. 点击 `文件` → `打开 Notebook`
3. 选择 `GitHub` 标签
4. 输入仓库：`liuchzzyy/echemistpy`
5. 选择分支：`cl_version`
6. 选择要打开的 notebook

## 🔧 运行说明

### 第一次运行

所有 notebook 的**第一个代码单元格**会自动：

1. ✅ 检测 Colab 环境
2. ✅ 从 GitHub 克隆项目代码
3. ✅ 安装所有依赖（xarray, scipy, scikit-learn 等）
4. ✅ 安装 echemistpy 包

**运行时间**: 约 1-2 分钟

### 数据下载

测试数据会**自动从 OneDrive 下载**：

- **数据大小**: 约 830 MB（STXM/TXM 数据）
- **下载时间**: 约 3-5 分钟
- **存储位置**: Colab 运行时环境（会话结束后删除）

**无需手动上传大文件！**

### 完整运行流程

```python
# 1. 环境设置（第 2 个单元格）
# 克隆仓库 + 安装依赖
# 运行时间: ~2 分钟

# 2. 导入模块（第 4 个单元格）
# 导入 echemistpy 模块
# 运行时间: ~5 秒

# 3. 下载数据（第 5 个单元格）
# 从 OneDrive 自动下载
# 运行时间: ~5 分钟

# 4. 加载数据（第 6 个单元格）
# 读取 HDF5 文件
# 运行时间: ~10 秒

# 5. 执行分析（第 7 个单元格）
# PCA + 聚类 + 拟合
# 运行时间: ~1 分钟

# 6. 可视化结果（第 8 个单元格）
# 绘制图表
# 运行时间: ~5 秒
```

## 💾 数据管理

### OneDrive 数据源

测试数据托管在 OneDrive/SharePoint：

- **URL**: https://uab-my.sharepoint.com/:u:/g/personal/1615992_uab_cat/IQCiwUxTb7I-QpG_3-5KDu3VAZQiEam_jrJOLEVC0rDR6vk
- **文件**: `20230629_E1A_749.7x177.5y_specnorm_aliOF.hdf5.hdf5`
- **大小**: 830.21 MB
- **类型**: STXM/TXM 数据（αMnO2 样品）

### 本地数据路径

下载后的数据位于：

```
/content/echemistpy/docs/examples/TXM/αMnO2/20230629_E1A_749.7x177.5y_specnorm_aliOF.hdf5.hdf5
```

### 使用自己的数据

如果要使用自己的数据文件：

#### 方法 1: Google Drive（推荐用于大文件）

```python
from google.colab import drive
drive.mount('/content/drive')

# 使用 Drive 中的文件
file_path = '/content/drive/MyDrive/my_data/sample.hdf5'
raw_data, raw_info = load(file_path)
```

#### 方法 2: 直接上传（< 100MB）

```python
from google.colab import files
uploaded = files.upload()  # 选择文件

# 使用上传的文件
for filename in uploaded.keys():
    raw_data, raw_info = load(filename)
```

#### 方法 3: 公开 URL 下载

```python
import urllib.request

url = "https://your-server.com/data.hdf5"
urllib.request.urlretrieve(url, "data.hdf5")

raw_data, raw_info = load("data.hdf5")
```

## ⚠️ 常见问题

### Q1: 下载数据失败

```
✗ 下载失败: HTTP Error 403: Forbidden
```

**原因**: OneDrive 链接可能过期或权限变更

**解决方案**:

1. 检查链接是否有效（在浏览器中打开）
2. 联系数据提供者更新链接
3. 使用备用方案（手动上传或 Google Drive）

### Q2: 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**:

- Colab 免费版: 12GB RAM
- 减少 `pca_components` 参数
- 使用 Colab Pro（更多内存）

### Q3: 运行时断开连接

Colab 免费版会话限制：

- 空闲超时: 90 分钟
- 最大运行时间: 12 小时

**解决方案**:

- 定期执行代码保持活跃
- 保存中间结果到 Google Drive
- 使用 Colab Pro

### Q4: 如何保存结果？

```python
# 保存到 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 保存图表
import matplotlib.pyplot as plt
plt.savefig('/content/drive/MyDrive/results/plot.png')

# 保存数据
result_data.data.to_netcdf('/content/drive/MyDrive/results/analysis.nc')

# 下载到本地
from google.colab import files
files.download('plot.png')
```

## 🔗 相关资源

- **项目主页**: https://github.com/liuchzzyy/echemistpy
- **完整文档**: [docs/](../docs/)
- **数据管理**: [docs/DATA_MANAGEMENT.md](../docs/DATA_MANAGEMENT.md)
- **开发指南**: [AGENTS.md](../AGENTS.md)

## 💡 提示

- 首次运行需要下载依赖和数据，请耐心等待
- 数据下载进度会实时显示
- 运行完毕后，Colab 会话数据会被清除（不影响 Google Drive）
- 建议将结果保存到 Google Drive 以持久化

---

**问题反馈**: [GitHub Issues](https://github.com/liuchzzyy/echemistpy/issues)
