"""Electrochemical analysis helpers."""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr
from traitlets import Bool, Float, Unicode
from traitlets import List as TList

from echemistpy.io.structures import RawData

from .registry import TechniqueAnalyzer


class GalvanostaticAnalyzer(TechniqueAnalyzer):
    """Analyze galvanostatic (constant-current) experiments.

    Produces capacity (cumulative charge) vs time, start/end/average potentials,
    and a normalized potential for visualization.
    """

    technique = Unicode("echem", help="Technique identifier")
    supported_techniques = TList(Unicode(), default_value=["echem", "gpcl", "gcd"], help="List of supported technique identifiers")

    # 数据列配置
    time_columns = TList(Unicode(), default_value=["time_s", "systime"], help="Candidate time column names in order of preference")
    potential_columns = TList(Unicode(), default_value=["ewe_v", "voltage_v"], help="Candidate potential/voltage column names in order of preference")
    current_columns = TList(Unicode(), default_value=["current_ma", "current_ua"], help="Candidate current column names in order of preference")

    # 分析选项
    calculate_ce = Bool(True, help="Whether to calculate coulombic efficiency if cycle_number is present")
    baseline_correct = Bool(True, help="Whether to compute baseline-corrected potential")

    # 电荷容量计算参数
    time_unit_conversion = Float(3600.0, help="Conversion factor from time unit to hours (default: 3600 for seconds to hours)")

    @staticmethod
    def _pick(ds: xr.Dataset, candidates: list[str]) -> str | None:
        """从数据集中选择第一个存在的列名.

        数据已经在 io/standardizer.py 中标准化, 直接查找标准列名.

        Args:
            ds: xarray Dataset
            candidates: 候选列名列表

        Returns:
            第一个找到的列名, 如果都不存在则返回 None
        """
        available = set(ds.data_vars) | set(ds.coords)
        for c in candidates:
            if c in available:
                return c
        return None

    @staticmethod
    def _get_column_candidates(trait_list: Any) -> list[str]:
        """将 traitlets List 转换为普通 list[str] 供类型检查使用.

        Args:
            trait_list: TList[Unicode] trait

        Returns:
            list[str]
        """
        # 在运行时, TList 就是 list, 但类型检查器需要显式转换
        return list(trait_list) if trait_list else []

    def validate(self, raw_data: RawData) -> None:
        """验证数据是否适合此分析器.

        要求:
        1. 数据的 technique 必须包含 "echem"
        2. 其他技术标识符 (如有) 应在 ["gpcl", "gcd"] 中
        """
        if not hasattr(raw_data, "info") or not hasattr(raw_data.info, "technique"):
            return  # 没有技术信息时跳过验证

        data_techniques = raw_data.info.technique

        # 统一转换为列表处理
        if not isinstance(data_techniques, list):
            data_techniques = [data_techniques] if data_techniques else []

        # 必须包含 "echem"
        if "echem" not in data_techniques:
            raise ValueError(f"GalvanostaticAnalyzer 要求数据技术必须包含 'echem', 但得到: {data_techniques}")

        # 其他技术标识符必须在允许列表中
        allowed_others = {"gpcl", "gcd"}  # 使用集合提高查找效率
        invalid_techniques = []
        for t in data_techniques:
            if t != "echem" and t not in allowed_others:
                invalid_techniques.append(t)

        if invalid_techniques:
            raise ValueError(f"数据包含不支持的技术标识符: {invalid_techniques}. 除 'echem' 外, 仅支持: {list(allowed_others)}")

    def preprocess(self, raw_data: RawData) -> RawData:
        """按时间排序并计算基线校正的电位."""
        if raw_data.is_tree:
            ds = raw_data.data.dataset
            if ds is None:
                raise ValueError("DataTree has no root dataset for galvanostatic analysis.")
        else:
            ds = raw_data.data

        # 使用 traitlets 配置的列名
        time_key = self._pick(ds, self._get_column_candidates(self.time_columns))
        pot_key = self._pick(ds, self._get_column_candidates(self.potential_columns))
        cur_key = self._pick(ds, self._get_column_candidates(self.current_columns))

        if time_key is None:
            raise ValueError(f"No time column found. Searched for: {self.time_columns}")
        if pot_key is None:
            raise ValueError(f"No potential/voltage column found. Searched for: {self.potential_columns}")
        if cur_key is None:
            raise ValueError(f"No current column found. Searched for: {self.current_columns}")

        # 按时间排序
        with contextlib.suppress(Exception):
            ds = ds.sortby(time_key)

        dim = ds[cur_key].dims[0]

        # 提取数值型时间数组
        t_vals = ds[time_key].values
        if np.issubdtype(getattr(t_vals, "dtype", object), np.datetime64):
            t_pd = pd.to_datetime(t_vals)
            t_numeric = (t_pd - t_pd[0]).total_seconds()
        else:
            try:
                t_numeric = np.asarray(t_vals, dtype=float)
            except Exception:
                t_numeric = np.arange(ds.sizes[dim], dtype=float)

        # 基线校正电位 (如果启用)
        if self.baseline_correct:
            potential_vals = ds[pot_key].values
            # 以初始值为参考, 突出变化
            ref = float(potential_vals[0]) if getattr(potential_vals, "size", 0) else 0.0
            baseline_corrected = potential_vals - ref
            raw_data.data = ds.assign(**{"baseline_corrected_potential": (dim, baseline_corrected)})
        else:
            raw_data.data = ds

        # 如果不存在 time_s 坐标, 则存储数值型时间
        if "time_s" not in raw_data.data.coords:
            raw_data.data = raw_data.data.assign_coords(time_s=(dim, t_numeric))

        return raw_data

    @staticmethod
    def split_by_cycle(ds: xr.Dataset) -> dict[int, xr.Dataset]:
        """将数据按照循环次数分割.

        Args:
            ds: 包含 cycle_number 的数据集

        Returns:
            字典, 键为循环编号, 值为对应的数据集
        """
        if "cycle_number" not in ds.coords and "cycle_number" not in ds.data_vars:
            raise ValueError("数据集中未找到 cycle_number 列, 无法按循环分割")

        cycle_numbers = ds["cycle_number"].values
        unique_cycles = np.unique(cycle_numbers)

        cycles = {}
        for cycle_id in unique_cycles:
            cycle_int = int(cycle_id)
            mask = cycle_numbers == cycle_id
            cycles[cycle_int] = ds.isel({ds.dims[0]: mask})

        return cycles

    def calc_CE(self, ds: xr.Dataset) -> pd.DataFrame:  # noqa: N802
        """计算每个循环的库伦效率.

        根据电流方向判断充电和放电过程, 计算库伦效率 = 放电容量 / 充电容量 * 100%

        Args:
            ds: 包含 cycle_number, current_ma 和 time_s 的数据集

        Returns:
            DataFrame, 包含每个循环的充电容量, 放电容量和库伦效率
        """
        if "cycle_number" not in ds.coords and "cycle_number" not in ds.data_vars:
            raise ValueError("数据集中未找到 cycle_number 列")

        # 使用 traitlets 配置的列名查找
        cur_candidates = self._get_column_candidates(self.current_columns)
        cur_key = self._pick(ds, cur_candidates)
        if cur_key is None:
            raise ValueError(f"未找到电流列. 搜索了: {cur_candidates}")

        time_candidates = self._get_column_candidates(self.time_columns)
        time_key = self._pick(ds, time_candidates)
        if time_key is None:
            raise ValueError(f"未找到时间列. 搜索了: {time_candidates}")

        # 按循环分割
        cycles = self.split_by_cycle(ds)

        results = []
        for cycle_num, cycle_ds in cycles.items():
            current = cycle_ds[cur_key].values

            # 获取时间
            time = cycle_ds[time_key].values
            if np.issubdtype(getattr(time, "dtype", object), np.datetime64):
                t_pd = pd.to_datetime(time)
                time_numeric = (t_pd - t_pd[0]).total_seconds()
            else:
                time_numeric = np.asarray(time, dtype=float)

            dt = np.gradient(time_numeric)

            # 计算电荷量, 使用配置的时间单位转换
            # mAh = mA * h = mA * s / time_unit_conversion
            charge = current * dt / self.time_unit_conversion

            # 充电为正电流, 放电为负电流 (根据实际情况可能需要调整符号)
            charge_capacity = np.sum(charge[charge > 0])  # 充电容量
            discharge_capacity = np.abs(np.sum(charge[charge < 0]))  # 放电容量

            # 库伦效率
            coulombic_eff = (discharge_capacity / charge_capacity) * 100 if charge_capacity > 0 else 0.0

            results.append({
                "cycle_number": cycle_num,
                "charge_capacity_mah": charge_capacity,
                "discharge_capacity_mah": discharge_capacity,
                "coulombic_efficiency_%": coulombic_eff,
            })

        return pd.DataFrame(results)

    def compute(self, raw_data: RawData) -> tuple[Dict[str, Any], Dict[str, xr.Dataset]]:
        """计算累计电荷 (容量) 和电位统计."""
        ds = raw_data.data
        if isinstance(ds, xr.DataTree):
            ds = ds.dataset
            if ds is None:
                raise ValueError("DataTree has no root dataset for galvanostatic analysis.")

        # 使用 traitlets 配置的列名
        time_key = self._pick(ds, self._get_column_candidates(self.time_columns)) or "time_s"
        pot_candidates = [*self._get_column_candidates(self.potential_columns), "baseline_corrected_potential"]
        pot_key = self._pick(ds, pot_candidates) or "baseline_corrected_potential"
        cur_key = self._pick(ds, self._get_column_candidates(self.current_columns)) or next(iter(ds.data_vars))

        # 时间数组
        time = ds.coords[time_key].values if time_key in ds.coords else ds[time_key].values

        potential = ds[pot_key].values
        current = ds[cur_key].values

        dim = ds[cur_key].dims[0]

        # 计算时间步长
        if np.issubdtype(getattr(time, "dtype", object), np.datetime64):
            t_pd = pd.to_datetime(time)
            t_numeric = (t_pd - t_pd[0]).total_seconds()
        else:
            t_numeric = np.asarray(time, dtype=float)
        dt = np.gradient(t_numeric)

        # 累计电荷 (电流对时间的积分)
        cumulative_charge = np.cumsum(current * dt)

        # 统计信息
        start_potential = float(potential[0]) if potential.size else float("nan")
        end_potential = float(potential[-1]) if potential.size else float("nan")
        avg_potential = float(np.mean(potential)) if potential.size else float("nan")
        net_charge = float(cumulative_charge[-1]) if cumulative_charge.size else 0.0

        # 归一化电位用于绘图
        scale = np.max(np.abs(potential)) or 1.0
        normalized_pot = potential / scale

        table = ds.assign(
            normalized_potential=(dim, normalized_pot),
            cumulative_charge=(dim, cumulative_charge),
        )

        summary: Dict[str, Any] = {
            "start_potential": start_potential,
            "end_potential": end_potential,
            "average_potential": avg_potential,
            "net_charge": net_charge,
        }

        # 如果启用库伦效率计算且存在 cycle_number
        if self.calculate_ce and ("cycle_number" in ds.coords or "cycle_number" in ds.data_vars):
            try:
                coulombic_eff_df = self.calc_CE(ds)
                summary["coulombic_efficiency"] = coulombic_eff_df.to_dict("records")
            except Exception as e:
                # 如果计算失败, 记录警告但不中断分析
                warnings.warn(f"库伦效率计算失败: {e}", stacklevel=2)

        return summary, {"galvanostatic_trace": table}
