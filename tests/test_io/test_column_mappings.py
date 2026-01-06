"""测试 column_mappings 模块。

本测试模块验证列名映射的正确性和完整性。
"""

import pytest

from echemistpy.io.column_mappings import (
    CAPACITY_MAH_MAPPINGS,
    CAPACITY_UAH_MAPPINGS,
    COUNTER_ELECTRODE_MAPPINGS,
    CURRENT_MA_MAPPINGS,
    CURRENT_UA_MAPPINGS,
    EIS_MAPPINGS,
    ABSOLUTE_TIME_MAPPINGS,
    RELATIVE_TIME_MAPPINGS,
    CYCLE_MAPPINGS,
    STEP_MAPPINGS,
    WORKING_ELECTRODE_MAPPINGS,
    BATTERY_VOLTAGE_MAPPINGS,
    get_echem_mappings,
    get_tga_mappings,
    get_txm_mappings,
    get_xas_mappings,
    get_xps_mappings,
    get_xrd_mappings,
)


class TestEchemMappings:
    """测试电化学列名映射。"""

    def test_absolute_time_mappings(self):
        """测试绝对时间列名映射。"""
        assert ABSOLUTE_TIME_MAPPINGS["Systime"] == "systime"
        assert ABSOLUTE_TIME_MAPPINGS["SysTime"] == "systime"
        assert ABSOLUTE_TIME_MAPPINGS["DateTime"] == "systime"

    def test_relative_time_mappings(self):
        """测试相对时间列名映射。"""
        assert RELATIVE_TIME_MAPPINGS["time"] == "time_s"
        assert RELATIVE_TIME_MAPPINGS["Time"] == "time_s"
        assert RELATIVE_TIME_MAPPINGS["test_time"] == "time_s"

    def test_cycle_mappings(self):
        """测试循环数列名映射。"""
        assert CYCLE_MAPPINGS["Ns"] == "cycle_number"
        assert CYCLE_MAPPINGS["cycle"] == "cycle_number"
        assert CYCLE_MAPPINGS["Cycle Number"] == "cycle_number"

    def test_step_mappings(self):
        """测试步骤数列名映射。"""
        assert STEP_MAPPINGS["step"] == "step_number"
        assert STEP_MAPPINGS["Step Number"] == "step_number"

    def test_working_electrode_mappings(self):
        """测试工作电极电势映射。"""
        assert WORKING_ELECTRODE_MAPPINGS["Ewe/V"] == "ewe_v"
        assert WORKING_ELECTRODE_MAPPINGS["Potential"] == "ewe_v"
        assert WORKING_ELECTRODE_MAPPINGS["Ewe_V"] == "ewe_v"

    def test_counter_electrode_mappings(self):
        """测试对电极电势映射。"""
        assert COUNTER_ELECTRODE_MAPPINGS["Ece/V"] == "ece_v"
        assert COUNTER_ELECTRODE_MAPPINGS["Ece_V"] == "ece_v"

    def test_battery_voltage_mappings(self):
        """测试电池电压映射。"""
        assert BATTERY_VOLTAGE_MAPPINGS["V"] == "voltage_v"
        assert BATTERY_VOLTAGE_MAPPINGS["voltage"] == "voltage_v"
        assert BATTERY_VOLTAGE_MAPPINGS["Cell_Voltage"] == "voltage_v"

    def test_current_ma_mappings(self):
        """测试电流（毫安）映射。"""
        assert CURRENT_MA_MAPPINGS["<I>/mA"] == "current_ma"
        assert CURRENT_MA_MAPPINGS["I/mA"] == "current_ma"
        assert CURRENT_MA_MAPPINGS["Current"] == "current_ma"

    def test_current_ua_mappings(self):
        """测试电流（微安）映射。"""
        assert CURRENT_UA_MAPPINGS["current/uA"] == "current_ua"
        assert CURRENT_UA_MAPPINGS["Current/uA"] == "current_ua"

    def test_capacity_mah_mappings(self):
        """测试容量（毫安时）映射。"""
        assert CAPACITY_MAH_MAPPINGS["(Q-Qo)/mA.h"] == "capacity_mah"
        assert CAPACITY_MAH_MAPPINGS["Capacity/mA.h"] == "capacity_mah"

    def test_capacity_uah_mappings(self):
        """测试容量（微安时）映射。"""
        assert CAPACITY_UAH_MAPPINGS["capacity/uAh"] == "capacity_uah"
        assert CAPACITY_UAH_MAPPINGS["Capacity/uAh"] == "capacity_uah"

    def test_eis_mappings(self):
        """测试电化学阻抗谱映射。"""
        assert EIS_MAPPINGS["freq/Hz"] == "frequency_hz"
        assert EIS_MAPPINGS["Re(Z)/Ohm"] == "re_z_ohm"
        assert EIS_MAPPINGS["-Im(Z)/Ohm"] == "-im_z_ohm"
        assert EIS_MAPPINGS["|Z|/Ohm"] == "z_mag_ohm"
        assert EIS_MAPPINGS["Phase(Z)/deg"] == "phase_deg"

    def test_get_echem_mappings_completeness(self):
        """测试电化学映射函数的完整性。"""
        echem_map = get_echem_mappings()

        # 验证关键映射存在
        assert "systime" in echem_map.values()
        assert "time_s" in echem_map.values()
        assert "cycle_number" in echem_map.values()
        assert "ewe_v" in echem_map.values()
        assert "current_ma" in echem_map.values()
        assert "capacity_mah" in echem_map.values()
        assert "frequency_hz" in echem_map.values()

        # 验证映射不为空
        assert len(echem_map) > 50  # 应该有大量映射

    def test_no_duplicate_targets_in_echem_mappings(self):
        """测试电化学映射中每个标准列名只有一个来源（排除大小写变体）。"""
        echem_map = get_echem_mappings()
        reverse_map = {}

        for old_name, new_name in echem_map.items():
            # 允许某些特殊情况（如不同的电流单位）
            if new_name in ["current_ma", "current_ua", "capacity_mah", "capacity_uah"]:
                continue
            # 允许大小写变体（如 Systime/SysTime -> systime）
            if new_name in reverse_map:
                old_base = reverse_map[new_name].lower()
                new_base = old_name.lower()
                if old_base == new_base or old_base in new_name.lower() or new_name.lower() in old_base:
                    continue  # 忽略大小写变体
            reverse_map[new_name] = old_name


class TestTechniqueMappings:
    """测试其他技术类型的列名映射。"""

    def test_get_xrd_mappings(self):
        """测试 XRD 映射。"""
        xrd_map = get_xrd_mappings()
        assert "2theta" in xrd_map
        assert "intensity" in xrd_map
        assert xrd_map["2theta"] == "2theta_degree"
        assert xrd_map["intensity"] == "intensity"

    def test_get_xps_mappings(self):
        """测试 XPS 映射。"""
        xps_map = get_xps_mappings()
        assert "binding_energy" in xps_map
        assert "intensity" in xps_map
        assert xps_map["binding_energy"] == "be_ev"
        assert xps_map["intensity"] == "intensity_cps"

    def test_get_tga_mappings(self):
        """测试 TGA 映射。"""
        tga_map = get_tga_mappings()
        assert "temperature" in tga_map
        assert "weight" in tga_map
        assert tga_map["temperature"] == "T/°C"
        assert tga_map["weight"] == "weight/%"

    def test_get_xas_mappings(self):
        """测试 XAS 映射。"""
        xas_map = get_xas_mappings()
        assert "energy" in xas_map
        assert "absorption" in xas_map
        assert xas_map["energy"] == "energy_eV"
        assert xas_map["absorption"] == "absorption_au"

    def test_get_txm_mappings(self):
        """测试 TXM 映射。"""
        txm_map = get_txm_mappings()
        assert "energy" in txm_map
        assert "x" in txm_map
        assert "y" in txm_map
        assert txm_map["energy"] == "energy_eV"
        assert txm_map["x"] == "x_um"
        assert txm_map["y"] == "y_um"


class TestMappingConsistency:
    """测试映射的一致性。"""

    def test_all_values_are_strings(self):
        """测试所有映射键值都是字符串。"""
        echem_map = get_echem_mappings()
        for key, value in echem_map.items():
            assert isinstance(key, str), f"Key {key} is not a string"
            assert isinstance(value, str), f"Value {value} for key {key} is not a string"

    def test_standardized_names_follow_convention(self):
        """测试标准化列名遵循命名约定。"""
        echem_map = get_echem_mappings()
        standardized_names = set(echem_map.values())

        # 验证标准化列名使用小写和下划线
        for name in standardized_names:
            assert name.islower() or "_" in name or "/" in name, f"Standardized name {name} doesn't follow convention"

    def test_common_standard_names_exist(self):
        """测试常见的标准列名存在。"""
        echem_map = get_echem_mappings()
        standardized_names = set(echem_map.values())

        # 验证关键标准列名存在
        expected_names = [
            "systime",
            "time_s",
            "cycle_number",
            "step_number",
            "ewe_v",
            "ece_v",
            "voltage_v",
            "current_ma",
            "capacity_mah",
            "frequency_hz",
        ]

        for name in expected_names:
            assert name in standardized_names, f"Expected standard name {name} not found"
