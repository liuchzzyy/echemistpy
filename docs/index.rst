.. echemistpy documentation master file

echemistpy 文档
================

欢迎使用 **echemistpy** 文档！这是一个用于电化学技术和材料表征（XRD、XPS、TGA、XAS、TXM）的统一数据处理框架。

.. note::
   此项目目前处于 **Alpha** 阶段，设计可能会在版本之间发生变化。

主要特性
--------

* **统一数据格式**：使用 `xarray.Dataset`（扁平数据）和 `xarray.DataTree`（分层数据）
* **可扩展的插件接口**：支持不同仪器和文件格式的读取器
* **模块化分析器模板**：基于 `TechniqueAnalyzer` 基类
* **管道编排**：使用 `AnalysisPipeline` 进行批处理
* **类型安全配置**：使用 traitlets 进行元数据管理

快速开始
========

安装
------

.. code-block:: bash

   pip install echemistpy

基本使用
--------

.. code-block:: python

   from echemistpy.io import load

   # 自动检测格式并加载数据
   raw_data, raw_info = load("data.mpt", sample_name="MySample")

   # 查看数据
   print(raw_data.data)

   # 查看元数据
   print(raw_info.to_dict())

支持的格式
============

电化学
---------

* **BioLogic** (.mpt) - 使用 ``BiologicMPTReader``
* **LANHE** (.xlsx) - 使用 ``LanheXLSXReader``

材料表征
---------

* **XRD** - MSPD (.xye) 使用 ``MSPDReader``
* **XAS** - CLAESS (.dat) 使用 ``CLAESSReader``
* **TXM** - MISTRAL (.hdf5) 使用 ``MISTRALReader``

文档内容
========

.. toctree::
   :maxdepth: 2

   api/io
   api/processing
   user_guide/index
   developer_guide/index

索引和表格
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
