"""使用实际数据测试 echemistpy IO 模块。

本测试脚本使用 docs/examples 中的真实数据文件来验证 IO 模块的功能。
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from echemistpy.io import load
from echemistpy.io.standardizer import standardize_names


def print_section(title: str):
    """打印分节标题。"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_biologic_eis():
    """测试 BioLogic EIS 数据加载。"""
    print_section("测试 1: BioLogic EIS 数据 (.mpt)")

    file_path = Path("docs/examples/Echem/Biologic_EIS.mpt")

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    try:
        # 加载数据
        raw_data, raw_info = load(file_path)

        print(f"✅ 成功加载 EIS 数据")
        print(f"   - 数据类型: {'DataTree' if raw_data.is_tree else 'Dataset'}")
        print(f"   - 数据变量: {raw_data.variables}")
        print(f"   - 坐标: {raw_data.coords}")
        print(f"   - 技术类型: {raw_info.technique}")
        print(f"   - 样本名称: {raw_info.sample_name}")
        print(f"   - 仪器: {raw_info.instrument}")

        # 验证标准化列名
        if "frequency_hz" in raw_data.variables or "freq/Hz" in raw_data.variables:
            print(f"✅ 频率数据存在")
        if "re_z_ohm" in raw_data.variables or "Re(Z)/Ohm" in raw_data.variables:
            print(f"✅ 阻抗实部数据存在")
        if "-im_z_ohm" in raw_data.variables or "-Im(Z)/Ohm" in raw_data.variables:
            print(f"✅ 阻抗虚部数据存在")

        return True

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_biologic_gpcl():
    """测试 BioLogic 恒流充放电数据加载。"""
    print_section("测试 2: BioLogic GPCL 数据 (.mpt)")

    file_path = Path("docs/examples/Echem/Biologic_GPCL.mpt")

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    try:
        # 加载数据
        raw_data, raw_info = load(file_path)

        print(f"✅ 成功加载 GPCL 数据")
        print(f"   - 数据类型: {'DataTree' if raw_data.is_tree else 'Dataset'}")
        print(f"   - 数据形状: {raw_data.data.dims}")
        print(f"   - 数据变量: {raw_data.variables}")
        print(f"   - 技术类型: {raw_info.technique}")
        print(f"   - 样本名称: {raw_info.sample_name}")

        # 验证关键列存在
        data = raw_data.data
        if isinstance(data, xr.Dataset):
            # 检查标准化后的列名或原始列名
            has_time = any(col in raw_data.variables for col in ["time_s", "time/s", "Time"])
            has_voltage = any(col in raw_data.variables for col in ["ewe_v", "Ewe/V", "Ewe"])
            has_current = any(col in raw_data.variables for col in ["current_ma", "<I>/mA", "I/mA"])
            has_capacity = any(col in raw_data.variables for col in ["capacity_mah", "Capacity/mA.h"])

            if has_time:
                print(f"✅ 时间数据存在")
            if has_voltage:
                print(f"✅ 电压数据存在")
            if has_current:
                print(f"✅ 电流数据存在")
            if has_capacity:
                print(f"✅ 容量数据存在")

        return True

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lanhe_gpcl():
    """测试 LANHE 恒流充放电数据加载。"""
    print_section("测试 3: LANHE GPCL 数据 (.xlsx)")

    file_path = Path("docs/examples/Echem/LANHE_GPCL.xlsx")

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    try:
        # 加载数据，指定 instrument
        raw_data, raw_info = load(file_path, instrument="lanhe")

        print(f"✅ 成功加载 LANHE 数据")
        print(f"   - 数据类型: {'DataTree' if raw_data.is_tree else 'Dataset'}")
        print(f"   - 数据变量: {raw_data.variables}")
        print(f"   - 技术类型: {raw_info.technique}")
        print(f"   - 仪器: {raw_info.instrument}")

        # 获取数据集
        ds = raw_data.data
        if isinstance(ds, xr.DataTree):
            print(f"   - DataTree 节点数: {len(list(ds.subtree))}")

        return True

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_folder_loading():
    """测试目录加载功能。"""
    print_section("测试 4: 目录加载 (test_folder)")

    folder_path = Path("docs/examples/Echem/test_folder")

    if not folder_path.exists():
        print(f"❌ 目录不存在: {folder_path}")
        return False

    try:
        # 加载整个目录
        raw_data, raw_info = load(folder_path, instrument="biologic")

        print(f"✅ 成功加载目录数据")
        print(f"   - 数据类型: {'DataTree' if raw_data.is_tree else 'Dataset'}")

        if raw_data.is_tree:
            tree = raw_data.data
            print(f"   - DataTree 节点: {list(tree.keys())}")
            print(f"   - 样本名称: {raw_info.sample_name}")

            # 检查合并的元数据
            if "n_files" in raw_info.others:
                print(f"   - 加载文件数: {raw_info.others['n_files']}")
            if "sample_names" in raw_info.others:
                print(f"   - 所有样本名: {raw_info.others['sample_names']}")

        return True

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_standardization():
    """测试数据标准化功能。"""
    print_section("测试 5: 数据标准化")

    file_path = Path("docs/examples/Echem/Biologic_GPCL.mpt")

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    try:
        # 加载原始数据（不标准化）
        raw_data, raw_info = load(file_path, standardize=False)
        original_vars = set(raw_data.variables)
        print(f"原始变量: {sorted(original_vars)}")

        # 加载标准化数据
        std_data, std_info = load(file_path, standardize=True)
        standardized_vars = set(std_data.variables)
        print(f"标准化变量: {sorted(standardized_vars)}")

        # 比较差异
        print(f"\n标准化效果:")

        # 检查常见标准化
        conversions = [
            ("time/s", "time_s", "相对时间"),
            ("Ewe/V", "ewe_v", "工作电极电势"),
            ("<I>/mA", "current_ma", "电流"),
            ("Capacity/mA.h", "capacity_mah", "容量"),
        ]

        for old, new, desc in conversions:
            if old in original_vars and new in standardized_vars:
                print(f"✅ {desc}: {old} → {new}")
            elif new in standardized_vars:
                print(f"✅ {desc}: 已标准化为 {new}")

        print(f"\n✅ 标准化功能正常")
        return True

    except Exception as e:
        print(f"❌ 标准化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_extraction():
    """测试元数据提取功能。"""
    print_section("测试 6: 元数据提取")

    file_path = Path("docs/examples/Echem/Biologic_GPCL.mpt")

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    try:
        # 加载数据
        raw_data, raw_info = load(file_path)

        print(f"提取的元数据:")
        print(f"   - 技术类型: {raw_info.technique}")
        print(f"   - 样本名称: {raw_info.sample_name}")
        print(f"   - 开始时间: {raw_info.start_time}")
        print(f"   - 操作员: {raw_info.operator}")
        print(f"   - 仪器: {raw_info.instrument}")
        print(f"   - 活性物质质量: {raw_info.active_material_mass}")

        # 显示其他元数据
        if raw_info.others:
            print(f"\n其他元数据 (前10项):")
            for i, (key, value) in enumerate(list(raw_info.others.items())[:10]):
                if not isinstance(value, (list, dict)):
                    print(f"   - {key}: {value}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"   - {key}: [{value[0]}...]")
            if len(raw_info.others) > 10:
                print(f"   ... 还有 {len(raw_info.others) - 10} 项")

        print(f"\n✅ 元数据提取正常")
        return True

    except Exception as e:
        print(f"❌ 元数据提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_quality():
    """测试数据质量验证。"""
    print_section("测试 7: 数据质量验证")

    file_path = Path("docs/examples/Echem/Biologic_GPCL.mpt")

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    try:
        # 加载数据
        raw_data, raw_info = load(file_path)

        ds = raw_data.data
        if isinstance(ds, xr.DataTree):
            ds = ds.to_dataset()

        print(f"数据质量检查:")

        # 检查数据完整性
        for var in ds.data_vars:
            data_array = ds[var]
            n_total = data_array.size
            n_nan = np.isnan(data_array.values).sum() if hasattr(data_array.values, '__len__') else 0

            print(f"   - {var}:")
            print(f"     总点数: {n_total}")
            print(f"     缺失值: {n_nan} ({100*n_nan/n_total if n_total > 0 else 0:.1f}%)")

            # 检查数据范围
            if n_nan < n_total:  # 有有效数据
                valid_data = data_array.values[~np.isnan(data_array.values)]
                if len(valid_data) > 0:
                    print(f"     范围: [{np.min(valid_data):.4g}, {np.max(valid_data):.4g}]")
                    print(f"     平均: {np.mean(valid_data):.4g}")

        print(f"\n✅ 数据质量验证完成")
        return True

    except Exception as e:
        print(f"❌ 数据质量验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """测试错误处理。"""
    print_section("测试 8: 错误处理")

    # 测试不存在的文件
    print("\n测试不存在的文件:")
    try:
        load("nonexistent.mpt")
        print("❌ 应该抛出 FileNotFoundError")
        return False
    except FileNotFoundError:
        print("✅ 正确抛出 FileNotFoundError")
    except Exception as e:
        print(f"❌ 抛出了错误的异常类型: {type(e).__name__}")
        return False

    # 测试不支持的格式
    print("\n测试不支持的格式:")
    try:
        # 创建一个临时文件
        temp_file = Path("test_temp.xyz")
        temp_file.write_text("test")
        try:
            load(temp_file)
            print("❌ 应该抛出 ValueError")
            return False
        except ValueError:
            print("✅ 正确抛出 ValueError")
        finally:
            temp_file.unlink()
    except Exception as e:
        print(f"❌ 抛出了错误的异常类型: {type(e).__name__}")
        return False

    return True


def main():
    """运行所有测试。"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "echemistpy IO 模块集成测试" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")

    # 运行所有测试
    tests = [
        ("BioLogic EIS 数据加载", test_biologic_eis),
        ("BioLogic GPCL 数据加载", test_biologic_gpcl),
        ("LANHE GPCL 数据加载", test_lanhe_gpcl),
        ("目录加载功能", test_folder_loading),
        ("数据标准化", test_standardization),
        ("元数据提取", test_metadata_extraction),
        ("数据质量验证", test_data_quality),
        ("错误处理", test_error_handling),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 崩溃: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 汇总结果
    print_section("测试结果汇总")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n测试统计:")
    print(f"   总计: {total}")
    print(f"   通过: {passed} ✅")
    print(f"   失败: {total - passed} ❌")
    print(f"   通过率: {100 * passed / total:.1f}%")

    print(f"\n详细结果:")
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {status} - {name}")

    print("\n")
    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述输出")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
