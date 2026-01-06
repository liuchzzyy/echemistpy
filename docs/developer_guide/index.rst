开发者指南
============

本指南面向希望为 echemistpy 做出贡献的开发者。

.. toctree::
   :maxdepth: 2

   architecture
   adding_readers
   adding_analyzers
   testing
   coding_standards

架构概览
==========

.. _architecture:

项目结构
----------

.. code-block:: text

   src/echemistpy/
   ├── io/                      # 数据输入/输出
   │   ├── structures.py        # 数据结构定义
   │   ├── plugin_manager.py    # 插件管理器
   │   ├── loaders.py           # 统一加载接口
   │   ├── standardizer.py      # 数据标准化
   │   ├── base_reader.py       # 读取器基类
   │   ├── reader_utils.py      # 工具函数
   │   ├── saver.py             # 数据保存
   │   ├── column_mappings.py   # 列名映射
   │   └── plugins/             # 读取器实现
   │       ├── echem/
   │       ├── xrd/
   │       ├── xas/
   │       └── txm/
   ├── processing/              # 数据处理
   │   └── analyzers/           # 分析器实现
   ├── pipelines/               # 管道编排
   └── utils/                   # 工具函数

数据流
------

.. code-block:: text

   原始文件 → IOPluginManager → RawData + RawDataInfo
                                     ↓
                             TechniqueAnalyzer
                                     ↓
                             ResultsData + ResultsDataInfo

插件系统
----------

echemistpy 使用插件系统来支持多种文件格式和仪器。

1. **插件注册**：在 ``IOPluginManager`` 中注册
2. **插件实现**：继承 ``BaseReader`` 类
3. **插件使用**：通过统一的 ``load()`` 接口

添加读取器
==========

.. _adding_readers:

创建新的读取器
---------------

1. **创建读取器类**：

.. code-block:: python

   from echemistpy.io.base_reader import BaseReader
   from echemistpy.io.structures import RawData, RawDataInfo

   class MyInstrumentReader(BaseReader):
       # 定义类变量
       supports_directories = True
       instrument = "my_instrument"

       def _load_single_file(self, path, **kwargs):
           # 实现文件读取逻辑
           import pandas as pd
           df = pd.read_csv(path)

           # 转换为 xarray.Dataset
           import xarray as xr
           ds = xr.Dataset.from_dataframe(df)

           # 创建元数据
           metadata = {"instrument": "my_instrument"}
           raw_info = RawDataInfo(
               sample_name=self.sample_name or path.stem,
               technique=["my_technique"],
               instrument=self.instrument,
               others=metadata
           )

           return RawData(data=ds), raw_info

2. **注册读取器**：

.. code-block:: python

   from echemistpy.io.plugin_manager import get_plugin_manager

   pm = get_plugin_manager()
   pm.register_loader([".myext"], MyInstrumentReader)

3. **测试读取器**：

.. code-block:: python

   from echemistpy.io import load

   raw_data, raw_info = load("data.myext")
   print(raw_data.data)

添加分析器
==========

.. _adding_analyzers:

创建新的分析器
---------------

1. **继承基类**：

.. code-block:: python

   from echemistpy.processing.analyzers.base import TechniqueAnalyzer

   class MyAnalyzer(TechniqueAnalyzer):
       technique = "my_technique"
       required_columns = ("column1", "column2")

       def compute(self, data, **kwargs):
           # 实现计算逻辑
           summary = {"mean": data.mean().values}
           results_table = {"processed": data.to_dataframe()}
           return summary, results_table

2. **验证数据**（可选）：

.. code-block:: python

   def validate(self, data):
       super().validate(data)
       # 添加自定义验证逻辑
       if data["column1"].max() > 100:
           raise ValueError("column1 values exceed threshold")

3. **预处理数据**（可选）：

.. code-block:: python

   def preprocess(self, data):
       data = super().preprocess(data)
       # 添加自定义预处理
       return data.dropna()

测试
======

.. _testing:

运行测试
--------

.. code-block:: bash

   # 运行所有测试
   pytest

   # 运行特定模块的测试
   pytest tests/io/

   # 生成覆盖率报告
   pytest --cov=echemistpy --cov-report=html

编写测试
----------

单元测试示例：

.. code-block:: python

   import pytest
   import numpy as np
   import xarray as xr

   def test_my_analyzer():
       # 创建测试数据
       data = xr.Dataset({
           "column1": ("x", [1, 2, 3, 4, 5]),
           "column2": ("x", [2, 4, 6, 8, 10])
       }, coords={"x": [0, 1, 2, 3, 4]})

       # 创建分析器实例
       analyzer = MyAnalyzer()

       # 验证数据
       analyzer.validate(data)

       # 计算结果
       summary, results_table = analyzer.compute(data)

       # 断言结果
       assert summary["mean"]["column1"] == 3.0

性能测试
----------

使用 pytest-benchmark：

.. code-block:: bash

   pip install pytest-benchmark
   pytest tests/io/benchmarks/test_standardizer_performance.py

代码规范
==========

.. _coding_standards:

代码风格
----------

使用 **Ruff** 进行代码检查和格式化：

.. code-block:: bash

   # 检查代码
   ruff check src/

   # 自动修复
   ruff check --fix src/

   # 格式化代码
   ruff format src()

类型检查
----------

使用 **ty** 进行类型检查：

.. code-block:: bash

   uv run ty check src/

提交规范
----------

提交消息必须使用中文，格式如下：

.. code-block:: text

   [类型] 简短描述

   详细说明（可选）

   类型可以是：
   - [FEATURE] 新功能
   - [FIX] Bug 修复
   - [DOCS] 文档更新
   - [REFACTOR] 代码重构
   - [TEST] 测试相关
   - [PERF] 性能优化

示例：

.. code-block:: text

   [FEATURE] 添加新的列名映射模块

   - 创建 column_mappings.py 模块
   - 将 STANDARD_MAPPINGS 拆分为按功能分组的映射字典
   - 添加工厂函数 get_xxx_mappings()
   - 提升可维护性和可扩展性
