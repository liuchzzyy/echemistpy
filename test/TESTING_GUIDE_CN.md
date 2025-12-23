# Testing Guide for echemistpy

本文档说明如何在 echemistpy 项目中运行和维护测试。

## 快速开始

### 安装测试依赖

```bash
pip install -e ".[test]"
```

### 运行所有测试

```bash
python -m pytest test/
```

### 运行测试并生成覆盖率报告

```bash
python -m pytest test/ --cov=src/echemistpy/io --cov-report=html
```

覆盖率报告会生成在 `htmlcov/` 目录下，打开 `htmlcov/index.html` 查看详细报告。

## 测试结构

### 测试文件组织

```
test/
├── conftest.py                    # 共享的 fixtures 和配置
├── README.md                      # 测试文档（英文）
├── test_io_structures.py          # 数据结构测试
├── test_io_loaders.py             # 数据加载测试
├── test_io_saver.py               # 数据保存测试
├── test_io_standardizer.py        # 数据标准化测试
└── test_io_plugin_manager.py     # 插件管理测试
```

### 当前测试统计

- **测试总数**: 54
- **代码覆盖率**: 78%
- **核心模块覆盖率**:
  - structures.py: 96%
  - saver.py: 96%
  - plugin_manager.py: 97%
  - standardizer.py: 92%
  - loaders.py: 88%

## 编写新测试

### 使用 fixtures

`conftest.py` 提供了多个有用的 fixtures：

```python
def test_example(sample_dataset, sample_raw_data_info):
    """测试示例"""
    data = RawData(data=sample_dataset)
    assert data.data.equals(sample_dataset)
```

### 测试命名规范

- 测试函数名以 `test_` 开头
- 使用描述性名称，清楚说明测试的内容
- 每个测试只测试一个功能点

### 测试数据文件

测试使用 `docs/examples/` 中的真实数据文件：

- `echem/Biologic_EIS.mpt` - BioLogic 电化学阻抗谱数据
- `echem/Biologic_GPCL.mpt` - BioLogic 恒流充放电数据
- `echem/LANHE_GPCL.xlsx` - LANHE 电池测试数据

## 常用命令

### 运行特定测试文件

```bash
python -m pytest test/test_io_structures.py -v
```

### 运行匹配特定模式的测试

```bash
python -m pytest test/ -k "standardizer" -v
```

### 显示测试输出

```bash
python -m pytest test/ -v -s
```

### 仅运行失败的测试

```bash
python -m pytest test/ --lf
```

### 生成 XML 格式的测试报告（用于 CI/CD）

```bash
python -m pytest test/ --junitxml=test-results.xml
```

## CI/CD 集成

### GitHub Actions 配置示例

在 `.github/workflows/test.yml` 中添加：

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -e ".[test]"
    
    - name: Run tests
      run: |
        python -m pytest test/ --cov=src/echemistpy/io --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

## 故障排除

### 缺少依赖

如果测试因缺少依赖而失败：

```bash
pip install -e ".[test]"
```

### 示例文件未找到

某些测试需要示例数据文件。如果文件缺失，这些测试会被跳过。查看跳过的测试：

```bash
python -m pytest test/ -v -rs
```

### 慢速测试

某些测试由于文件 I/O 操作可能较慢。可以标记慢速测试：

```python
@pytest.mark.slow
def test_large_file_processing():
    pass
```

然后排除慢速测试运行：

```bash
python -m pytest test/ -m "not slow"
```

## 最佳实践

1. **每次提交前运行测试**: 确保代码更改不会破坏现有功能
2. **编写新功能时同时编写测试**: 保持测试覆盖率
3. **使用描述性的测试名称**: 让测试失败时易于理解
4. **保持测试独立**: 每个测试应该能够独立运行
5. **使用 fixtures**: 避免重复的测试设置代码
6. **测试边界条件**: 不仅测试正常情况，也测试异常情况

## 贡献测试

如果您想为项目贡献测试：

1. Fork 项目
2. 创建测试分支
3. 编写测试（确保通过）
4. 提交 Pull Request
5. 确保 CI 测试通过

## 参考资料

- [pytest 文档](https://docs.pytest.org/)
- [pytest-cov 文档](https://pytest-cov.readthedocs.io/)
- [xarray 测试指南](https://docs.xarray.dev/en/stable/contributing.html#test-driven-development-code-writing)
