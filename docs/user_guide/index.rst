用户指南
========

本指南帮助您开始使用 echemistpy 进行数据处理。

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   data_loading
   data_standardization
   data_analysis

安装
====

.. _installation:

使用 pip 安装
-------------

.. code-block:: bash

   pip install echemistpy

从源码安装
-----------

.. code-block:: bash

   git clone https://github.com/liuchzzyy/echemistpy.git
   cd echemistpy
   pip install -e .

可选依赖
----------

开发版本：

.. code-block:: bash

   pip install echemistpy[dev]

文档版本：

.. code-block:: bash

   pip install echemistpy[docs]

交互式版本：

.. code-block:: bash

   pip install echemistpy[interactive]

快速开始
========

.. _quickstart:

加载数据
---------

.. code-block:: python

   from echemistpy.io import load

   # 自动检测格式
   raw_data, raw_info = load("data.mpt", sample_name="MySample")

   # 查看数据
   print(raw_data.data)

标准化数据
-----------

.. code-block:: python

   from echemistpy.io import load

   raw_data, raw_info = load("data.xlsx", instrument="lanhe")

   # 数据已自动标准化
   print(raw_data.data.variables)

数据加载
=========

.. _data_loading:

支持的格式
----------

.. list-table::
   :header-rows: 1

   * - 技术
     - 格式
     - 扩展名
     - 读取器
   * - 电化学
     - BioLogic EC-Lab
     - .mpt
     - ``BiologicMPTReader``
   * - 电化学
     - LANHE 电池测试
     - .xlsx
     - ``LanheXLSXReader``
   * - XRD
     - MSPD
     - .xye
     - ``MSPDReader``
   * - XAS
     - ALBA CLAESS
     - .dat
     - ``CLAESSReader``
   * - TXM
     - MISTRAL
     - .hdf5
     - ``MISTRALReader``

加载选项
----------

指定仪器：

.. code-block:: python

   raw_data, raw_info = load("data.xlsx", instrument="lanhe")

覆盖元数据：

.. code-block:: python

   raw_data, raw_info = load(
       "data.mpt",
       sample_name="Sample001",
       operator="张三",
       active_material_mass="10.5 mg"
   )

数据标准化
===========

.. _data_standardization:

标准列名
----------

echemistpy 将不同仪器的列名映射到统一的标准名称。

电化学数据：

.. list-table::
   :header-rows: 1

   * - 标准列名
     - 含义
     - 单位
   * - time_s
     - 相对时间
     - s
   * - systime
     - 绝对时间
     - datetime64
   * - cycle_number
     - 循环数
     -
   * - ewe_v
     - 工作电极电势
     - V
   * - current_ma
     - 电流
     - mA
   * - capacity_mah
     - 容量
     - mAh

自定义映射
-----------

.. code-block:: python

   from echemistpy.io import load
   from echemistpy.io.standardizer import standardize_names

   raw_data, raw_info = load("data.csv", standardize=False)

   # 自定义映射
   custom_mapping = {"Voltage": "custom_voltage"}
   standardized_data, standardized_info = standardize_names(
       raw_data,
       raw_info,
       custom_mapping=custom_mapping
   )

数据分析
=========

.. _data_analysis:

使用分析器
----------

.. code-block:: python

   from echemistpy.processing.analyzers.echem import GalvanostaticAnalyzer

   analyzer = GalvanostaticAnalyzer(
       charge_cutoff_voltage=4.2,
       discharge_cutoff_voltage=2.5
   )

   results_data, results_info = analyzer.compute(raw_data, raw_info)

管道处理
----------

.. code-block:: python

   from echemistpy.pipelines import AnalysisPipeline

   pipeline = AnalysisPipeline()

   # 批量处理
   for sample in samples:
       raw_data, raw_info = load(sample)
       results = pipeline.run(raw_data, raw_info)
