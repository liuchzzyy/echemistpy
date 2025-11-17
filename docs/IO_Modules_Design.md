# echemistpy I/O 模块设计说明

## 模块功能分工

### `loaders.py` - 通用数据加载器
**功能：读取所有类型的原始数据文件**

- **主要职责：**
  - 支持多种文件格式（CSV, TSV, JSON, Excel, HDF5, NetCDF 等）
  - 自动检测文件格式和分隔符
  - 统一返回 `xarray.Dataset` 格式
  - 提供文件信息查询功能

- **核心函数：**
  - `load_table()` - 基础表格文件加载
  - `load_data_file()` - 通用文件加载器（自动格式检测）
  - `get_file_info()` - 文件信息分析
  - `register_loader()` - 注册自定义加载器
  - `list_supported_formats()` - 列出支持的文件格式

### `organization.py` - 数据组织和标准化
**功能：选择有用数据，标准化列名和格式，为分析做准备**

- **主要职责：**
  - 数据清理（去重、异常值处理、缺失值填充）
  - 列名标准化（统一命名规范）
  - 单位转换（统一到标准单位）
  - 数据格式验证
  - 技术类型自动检测

- **核心类：**
  - `DataCleaner` - 基础数据清理功能
  - `DataStandardizer` - 数据标准化专用类

- **核心函数：**
  - `standardize_measurement()` - 一步式标准化
  - `detect_measurement_technique()` - 自动检测实验技术
  - `clean_measurement()` - 基础清理
  - `validate_measurement_integrity()` - 数据完整性验证

## 数据处理流程

### 1. 数据加载阶段 (`loaders.py`)
```python
# 加载任意格式的数据文件
from echemistpy.io import load_data_file, get_file_info

# 先查看文件信息
info = get_file_info("data.xlsx")
print(f"文件大小: {info['size_bytes']} bytes")
print(f"检测到的列: {info.get('columns', [])}")

# 加载数据 - 自动检测格式
dataset = load_data_file("data.xlsx")
print(f"数据形状: {dataset.sizes}")
print(f"列名: {list(dataset.data_vars.keys())}")
```

### 2. 数据组织阶段 (`organization.py`)
```python
# 创建 Measurement 对象
from echemistpy.io import Measurement, MeasurementMetadata, standardize_measurement

metadata = MeasurementMetadata(
    technique="cv",  # 或自动检测
    sample_name="Sample-001",
    instrument="BioLogic SP-300"
)
measurement = Measurement(data=dataset, metadata=metadata)

# 标准化处理
standardized = standardize_measurement(
    measurement,
    technique_hint="cv",  # 可选：指定技术类型
    custom_mapping={"Time_s": "time/s"},  # 可选：自定义列名映射
    required_columns=["time/s", "Ewe/V", "I/mA"]  # 可选：必需列
)
```

### 3. 高级数据组织功能
```python
from echemistpy.io import DataStandardizer

# 使用 DataStandardizer 进行精细控制
standardizer = DataStandardizer(measurement)

# 链式操作
result = (standardizer
          .standardize_column_names()  # 标准化列名
          .standardize_units()         # 单位转换 (mV→V, µA→mA等)
          .ensure_required_columns(['time/s', 'Ewe/V', 'I/mA'])
          .get_standardized_measurement())

# 验证结果
issues = standardizer.validate_data_format()
if not issues['errors']:
    print("✓ 数据已准备好进行分析")
```

## 标准化规则

### 电化学数据 (Electrochemistry)
- **时间列：** `"time/s"` (秒为单位)
- **电位列：** `"Ewe/V"` (工作电极电位，伏特)
- **电流列：** `"I/mA"` (毫安培)
- **电荷列：** `"Q/mA.h"` (毫安时)
- **功率列：** `"P/W"` (瓦特)

### XRD 数据
- **角度列：** `"2theta/deg"` (度)
- **强度列：** `"intensity/counts"` (计数)

### XPS 数据  
- **结合能：** `"BE/eV"` (电子伏特)
- **强度列：** `"intensity/cps"` (每秒计数)

### TGA 数据
- **温度列：** `"T/°C"` (摄氏度)
- **重量列：** `"weight/%"` (百分比)
- **时间列：** `"time/min"` (分钟)

## 自动单位转换

系统会自动进行以下单位转换：

- **时间：** 分钟 → 秒，小时 → 秒
- **电流：** 安培 → 毫安培，微安培 → 毫安培  
- **电压：** 毫伏 → 伏特

## 与分析模块的接口

标准化后的数据具备以下特点，可直接用于分析模块：

1. **统一的行维度：** 所有数据都使用 `"row"` 作为主维度
2. **标准化的列名：** 遵循 echemistpy 命名规范
3. **一致的单位：** 统一转换为标准单位
4. **完整的元数据：** 包含实验技术、样品信息等
5. **验证通过：** 确保数据格式符合分析要求

## 使用示例

参见 `examples/data_processing_demo.py` 获取完整的使用演示。

---

**设计原则：**
- `loaders.py`：专注于数据读取，保持原始数据完整性
- `organization.py`：专注于数据标准化，为分析做准备
- 两个模块配合使用，形成完整的"原始数据 → 分析就绪数据"流水线