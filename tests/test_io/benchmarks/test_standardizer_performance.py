"""DataStandardizer 的性能基准测试。

本测试模块验证标准化的性能是否满足要求。
使用 pytest-benchmark 进行性能测量。
"""

import numpy as np
import pytest
import xarray as xr

# 只有在安装了 pytest-benchmark 时才运行这些测试
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

from echemistpy.io.column_mappings import get_echem_mappings
from echemistpy.io.standardizer import DataStandardizer


@pytest.fixture
def small_dataset():
    """创建小型测试数据集（100 行）。"""
    n_rows = 100
    data = {
        "time/s": np.arange(n_rows, dtype=float),
        "Ewe/V": np.random.randn(n_rows) * 0.1 + 3.7,
        "<I>/mA": np.random.randn(n_rows) * 0.5 + 1.0,
        "Cycle Number": np.repeat([1, 2, 3, 4, 5], n_rows // 5),
        "record": np.arange(n_rows),
    }
    ds = xr.Dataset(data, coords={"record": np.arange(n_rows)})
    return ds


@pytest.fixture
def medium_dataset():
    """创建中型测试数据集（10,000 行）。"""
    n_rows = 10_000
    data = {
        "time/s": np.arange(n_rows, dtype=float),
        "Ewe/V": np.random.randn(n_rows) * 0.1 + 3.7,
        "<I>/mA": np.random.randn(n_rows) * 0.5 + 1.0,
        "Cycle Number": np.repeat(np.arange(100), n_rows // 100),
        "record": np.arange(n_rows),
    }
    ds = xr.Dataset(data, coords={"record": np.arange(n_rows)})
    return ds


@pytest.fixture
def large_dataset():
    """创建大型测试数据集（100,000 行）。"""
    n_rows = 100_000
    data = {
        "time/s": np.arange(n_rows, dtype=float),
        "Ewe/V": np.random.randn(n_rows) * 0.1 + 3.7,
        "<I>/mA": np.random.randn(n_rows) * 0.5 + 1.0,
        "Cycle Number": np.repeat(np.arange(1000), n_rows // 1000),
        "record": np.arange(n_rows),
    }
    ds = xr.Dataset(data, coords={"record": np.arange(n_rows)})
    return ds


class TestDataStandardizerPerformance:
    """测试 DataStandardizer 的性能。"""

    def test_standardize_column_names_small(self, benchmark, small_dataset):
        """测试小型数据集的列名标准化性能。"""

        def run_standardize():
            standardizer = DataStandardizer(small_dataset, techniques=["echem"])
            return standardizer.standardize_column_names()

        result = benchmark(run_standardize)
        assert result.dataset.data_vars is not None

    def test_standardize_column_names_medium(self, benchmark, medium_dataset):
        """测试中型数据集的列名标准化性能。"""

        def run_standardize():
            standardizer = DataStandardizer(medium_dataset, techniques=["echem"])
            return standardizer.standardize_column_names()

        result = benchmark(run_standardize)
        assert result.dataset.data_vars is not None

    def test_standardize_column_names_large(self, benchmark, large_dataset):
        """测试大型数据集的列名标准化性能。"""

        def run_standardize():
            standardizer = DataStandardizer(large_dataset, techniques=["echem"])
            return standardizer.standardize_column_names()

        result = benchmark(run_standardize)
        assert result.dataset.data_vars is not None

    def test_standardize_units_small(self, benchmark, small_dataset):
        """测试小型数据集的单位标准化性能。"""

        def run_standardize():
            standardizer = DataStandardizer(small_dataset, techniques=["echem"])
            return standardizer.standardize_units()

        result = benchmark(run_standardize)
        assert result.dataset.data_vars is not None

    def test_standardize_units_medium(self, benchmark, medium_dataset):
        """测试中型数据集的单位标准化性能。"""

        def run_standardize():
            standardizer = DataStandardizer(medium_dataset, techniques=["echem"])
            return standardizer.standardize_units()

        result = benchmark(run_standardize)
        assert result.dataset.data_vars is not None

    def test_standardize_units_large(self, benchmark, large_dataset):
        """测试大型数据集的单位标准化性能。"""

        def run_standardize():
            standardizer = DataStandardizer(large_dataset, techniques=["echem"])
            return standardizer.standardize_units()

        result = benchmark(run_standardize)
        assert result.dataset.data_vars is not None

    def test_full_standardize_small(self, benchmark, small_dataset):
        """测试小型数据集的完整标准化性能。"""

        def run_standardize():
            standardizer = DataStandardizer(small_dataset, techniques=["echem"])
            return standardizer.standardize()

        result = benchmark(run_standardize)
        assert result.dataset.data_vars is not None

    def test_full_standardize_medium(self, benchmark, medium_dataset):
        """测试中型数据集的完整标准化性能。"""

        def run_standardize():
            standardizer = DataStandardizer(medium_dataset, techniques=["echem"])
            return standardizer.standardize()

        result = benchmark(run_standardize)
        assert result.dataset.data_vars is not None

    def test_full_standardize_large(self, benchmark, large_dataset):
        """测试大型数据集的完整标准化性能。"""

        def run_standardize():
            standardizer = DataStandardizer(large_dataset, techniques=["echem"])
            return standardizer.standardize()

        result = benchmark(run_standardize)
        assert result.dataset.data_vars is not None


class TestMappingRetrievalPerformance:
    """测试映射检索的性能。"""

    def test_get_echem_mappings_performance(self, benchmark):
        """测试电化学映射检索性能。"""

        def get_mappings():
            return get_echem_mappings()

        mappings = benchmark(get_mappings)
        assert len(mappings) > 50

    def test_mapping_lookup_performance(self, benchmark):
        """测试映射查找性能。"""
        mappings = get_echem_mappings()
        test_keys = list(mappings.keys()) * 100  # 重复 100 次

        def lookup_mappings():
            for key in test_keys:
                _ = mappings[key]

        benchmark(lookup_mappings)


class TestStandardizeCorrectness:
    """验证标准化的正确性。"""

    def test_column_renamed_correctly(self, small_dataset):
        """测试列名正确重命名。"""
        standardizer = DataStandardizer(small_dataset, techniques=["echem"])
        result = standardizer.standardize()

        # 验证列已被重命名
        assert "time_s" in result.dataset.data_vars
        assert "ewe_v" in result.dataset.data_vars
        assert "current_ma" in result.dataset.data_vars
        assert "cycle_number" in result.dataset.data_vars

        # 验证旧列名不存在
        assert "time/s" not in result.dataset.data_vars
        assert "Ewe/V" not in result.dataset.data_vars
        assert "<I>/mA" not in result.dataset.data_vars

    def test_data_preserved_after_standardize(self, small_dataset):
        """测试标准化后数据保持不变。"""
        standardizer = DataStandardizer(small_dataset, techniques=["echem"])
        result = standardizer.standardize()

        # 验证数据行数保持不变
        assert len(result.dataset.coords["record"]) == len(small_dataset.coords["record"])

        # 验证数据值保持一致
        np.testing.assert_array_almost_equal(
            result.dataset["time_s"].values,
            small_dataset["time/s"].values,
        )
        np.testing.assert_array_almost_equal(
            result.dataset["ewe_v"].values,
            small_dataset["Ewe/V"].values,
        )
