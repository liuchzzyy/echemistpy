"""Data standardization utilities for echemistpy.

This module handles the conversion of raw measurement data into standardized
formats with consistent column names and units across different instruments
and techniques.
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from traitlets import HasTraits, Instance, List, Unicode

from echemistpy.io.structures import (
    RawData,
    RawDataInfo,
)


class DataStandardizer(HasTraits):
    """Standardize raw data to echemistpy summary format."""

    dataset = Instance(xr.Dataset, help="The dataset to standardize")
    techniques = List(Unicode(), help="List of technique identifiers (e.g., ['echem', 'peis'])")
    instrument = Unicode(None, allow_none=True, help="The instrument identifier")

    # Standard column name mappings for different techniques
    STANDARD_MAPPINGS: ClassVar[dict[str, dict[str, str]]] = {
        "echem": {
            # Absolute Time (No units)
            "Systime": "Systime",
            "SysTime": "Systime",
            "systime": "Systime",
            "abs_time": "Systime",
            "Absolute Time": "Systime",
            "DateTime": "Systime",
            # Relative Time (Standardized to Time/s)
            "time": "Time/s",
            "Time": "Time/s",
            "TIME": "Time/s",
            "t": "Time/s",
            "time/s": "Time/s",
            "time_s": "Time/s",
            "Time_s": "Time/s",
            "test_time": "Time/s",
            "TestTime": "Time/s",
            # Cycle variants
            "Ns": "Cycle_number",
            "ns": "Cycle_number",
            "cycle": "Cycle_number",
            "Cycle": "Cycle_number",
            "cycle number": "Cycle_number",
            "cycle_number": "Cycle_number",
            "step": "Step_number",
            "step_number": "Step_number",
            "record": "Record",
            "record_number": "Record",
            # Potential/Voltage variants (working electrode)
            "potential": "Ewe/V",
            "Potential": "Ewe/V",
            "E": "Ewe/V",
            "Ewe": "Ewe/V",
            "Ewe/V": "Ewe/V",
            "Ece/V": "Ece/V",
            "potential_V": "Ewe/V",
            "voltage_v": "Ewe/V",
            # Battery voltage variants (cell voltage)
            "voltage": "Voltage/V",
            "Voltage": "Voltage/V",
            "V": "Voltage/V",
            "voltage_V": "Voltage/V",
            "battery_voltage": "Voltage/V",
            "Battery_Voltage": "Voltage/V",
            "V_batt": "Voltage/V",
            "Vbatt": "Voltage/V",
            "cell_voltage": "Voltage/V",
            "Cell_Voltage": "Voltage/V",
            "voltage/V": "Voltage/V",
            "Voltage/V": "Voltage/V",
            # Current variants
            "current": "Current/mA",
            "Current": "Current/mA",
            "I": "Current/mA",
            "i": "Current/mA",
            "current_mA": "Current/mA",
            "I_mA": "Current/mA",
            "<I>/mA": "Current/mA",
            "I/mA": "Current/mA",
            "control/V/mA": "Current/mA",
            "current_ua": "Current/uA",
            "current_ma": "Current/mA",
            "current/uA": "Current/uA",
            "current/mA": "Current/mA",
            # Charge/Capacity variants
            "charge": "Q/mA.h",
            "Charge": "Q/mA.h",
            "Q": "Q/mA.h",
            "capacity": "Capacity/mAh",
            "Capacity": "Capacity/mAh",
            "(Q-Qo)/mA.h": "Capacity/mAh",
            "capacity_uah": "Capacity/uAh",
            "capacity_mah": "Capacity/mAh",
            "capacity/uAh": "Capacity/uAh",
            "capacity/mAh": "Capacity/mAh",
            "capacity/mA.h": "Capacity/mAh",
            "Capacity/mA.h": "Capacity/mAh",
            "capacity/uA.h": "Capacity/uAh",
            "Capacity/uA.h": "Capacity/uAh",
            "SpeCap/mAh/g": "Specific_Capacity/mAh/g",
            "SpeCap_cal/mAh/g": "Specific_Capacity_cal/mAh/g",
            "SpeCap_cal/mA.h/g": "Specific_Capacity_cal/mAh/g",
            "Capacity_mA.h": "Capacity/mAh",
            "SpeCap_cal_mAh_g": "Specific_Capacity_cal/mAh/g",
            # EIS variants
            "freq/Hz": "Frequency/Hz",
            "frequency": "Frequency/Hz",
            "Frequency": "Frequency/Hz",
            "Re(Z)/Ohm": "Re(Z)/Ohm",
            "Z'": "Re(Z)/Ohm",
            "Z_real": "Re(Z)/Ohm",
            "-Im(Z)/Ohm": "-Im(Z)/Ohm",
            "Z''": "-Im(Z)/Ohm",
            "Z_imag": "-Im(Z)/Ohm",
            "|Z|/Ohm": "|Z|/Ohm",
            "Z_mag": "|Z|/Ohm",
            "Phase(Z)/deg": "Phase/deg",
            "phase": "Phase/deg",
            "Phase": "Phase/deg",
        },
        "xrd": {
            "2theta": "2theta/deg",
            "2Theta": "2theta/deg",
            "angle": "2theta/deg",
            "intensity": "intensity",
            "Intensity": "intensity",
            "counts": "intensity",
            "Counts": "intensity",
            "d-spacing": "d_spacing/Å",
            "d_spacing": "d_spacing/Å",
        },
        "xps": {
            "binding_energy": "BE/eV",
            "BE": "BE/eV",
            "energy": "BE/eV",
            "intensity": "intensity/cps",
            "Intensity": "intensity/cps",
            "counts": "intensity/cps",
            "cps": "intensity/cps",
        },
        "tga": {
            "temperature": "T/°C",
            "Temperature": "T/°C",
            "T": "T/°C",
            "weight": "weight/%",
            "Weight": "weight/%",
            "mass": "weight/%",
            "time": "time/min",
            "Time": "time/min",
            "t": "time/min",
        },
    }

    def __init__(
        self,
        dataset: xr.Dataset,
        techniques: list[str] | str = "unknown",
        instrument: Optional[str] = None,
        **kwargs,
    ):
        """Initialize with a dataset, technique(s), and instrument."""
        if isinstance(techniques, str):
            techniques = [techniques]
        super().__init__(
            dataset=dataset.copy(deep=True),
            techniques=[t.lower() for t in techniques],
            instrument=instrument,
            **kwargs,
        )

    def standardize(self, custom_mapping: Optional[Dict[str, str]] = None) -> "DataStandardizer":
        """Perform full standardization (names and units).

        Args:
            custom_mapping: Optional additional column name mappings

        Returns:
            Self for chaining
        """
        return self.standardize_column_names(custom_mapping).standardize_units()

    def standardize_column_names(self, custom_mapping: Optional[Dict[str, str]] = None) -> "DataStandardizer":
        """Standardize column names based on techniques, instrument, and custom mappings."""
        # Build aggregate mapping from all techniques
        mapping = {}
        for tech in self.techniques:
            # Map specific techniques to general categories if needed
            tech_category = tech
            if tech in {
                "cv",
                "gcd",
                "eis",
                "ca",
                "cp",
                "lsjv",
                "echem",
                "ec",
                "peis",
                "gpcl",
                "ocv",
            }:
                tech_category = "echem"

            # 1. Apply general technique mapping
            if tech_category in self.STANDARD_MAPPINGS:
                mapping.update(self.STANDARD_MAPPINGS[tech_category])

            # 2. Apply instrument-specific mapping (overwrites general)
            if self.instrument:
                inst_key = f"{self.instrument.lower()}_{tech_category}"
                if inst_key in self.STANDARD_MAPPINGS:
                    mapping.update(self.STANDARD_MAPPINGS[inst_key])

        # Add custom mappings if provided
        if custom_mapping:
            mapping.update(custom_mapping)

        # Apply renaming
        rename_dict = {}
        # Check both data variables and coordinates
        all_names = list(self.dataset.data_vars) + list(self.dataset.coords)
        for old_name in all_names:
            if old_name in mapping:
                rename_dict[old_name] = mapping[old_name]

        if rename_dict:
            self.dataset = self.dataset.rename(rename_dict)

        return self

    def standardize_units(self) -> "DataStandardizer":
        """Convert units to standard echemistpy conventions."""
        for var_name in list(self.dataset.data_vars.keys()):
            var_data = self.dataset[var_name]

            # Handle time conversions
            if "time" in var_name.lower() or "Time" in var_name:
                if "min" in var_name:
                    # Convert minutes to seconds
                    self.dataset[var_name] = var_data * 60
                    new_name = var_name.replace("min", "s")
                    self.dataset = self.dataset.rename({var_name: new_name})
                elif "h" in var_name and "mA.h" not in var_name:
                    # Convert hours to seconds
                    self.dataset[var_name] = var_data * 3600
                    new_name = var_name.replace("h", "s")
                    self.dataset = self.dataset.rename({var_name: new_name})

            # Handle current conversions
            elif "current" in var_name.lower() or var_name.startswith("I"):
                if "/A" in var_name or "_A" in var_name:
                    # Convert A to mA
                    self.dataset[var_name] = var_data * 1000
                    new_name = var_name.replace("/A", "/mA").replace("_A", "_mA")
                    self.dataset = self.dataset.rename({var_name: new_name})
                elif "/μA" in var_name or "/uA" in var_name:
                    # Convert μA to mA
                    self.dataset[var_name] = var_data / 1000
                    new_name = var_name.replace("/μA", "/mA").replace("/uA", "/mA")
                    self.dataset = self.dataset.rename({var_name: new_name})

            # Handle voltage conversions
            elif ("voltage" in var_name.lower() or "potential" in var_name.lower() or var_name.startswith("E")) and "/mV" in var_name:
                # Convert mV to V
                self.dataset[var_name] = var_data / 1000
                new_name = var_name.replace("/mV", "/V")
                self.dataset = self.dataset.rename({var_name: new_name})

            # Handle capacity conversions
            elif ("capacity" in var_name.lower() or var_name.startswith("Q")) and "/uAh" in var_name:
                # Convert uAh to mAh
                self.dataset[var_name] = var_data / 1000
                new_name = var_name.replace("/uAh", "/mAh")
                self.dataset = self.dataset.rename({var_name: new_name})

        return self

    def ensure_required_columns(self, required_columns: List[str]) -> "DataStandardizer":
        """Ensure that required columns exist, creating placeholders if needed."""
        missing_cols = []
        for col in required_columns:
            if col not in self.dataset.data_vars:
                missing_cols.append(col)

        if missing_cols:
            warnings.warn(
                f"Missing required columns: {missing_cols}. Creating placeholders.",
                stacklevel=2,
            )
            # Create placeholder columns with NaN values
            if "record" in self.dataset.coords:
                n_rows = len(self.dataset.coords["record"])
                for col in missing_cols:
                    self.dataset[col] = ("record", np.full(n_rows, np.nan))
            elif "row" in self.dataset.coords:
                n_rows = len(self.dataset.coords["row"])
                for col in missing_cols:
                    self.dataset[col] = ("row", np.full(n_rows, np.nan))

        return self

    def get_dataset(self) -> xr.Dataset:
        """Return the standardized dataset."""
        return self.dataset


def standardize_names(
    raw_data: RawData,
    raw_data_info: RawDataInfo,
    technique_hint: Optional[str | list[str]] = None,
    custom_mapping: Optional[Dict[str, str]] = None,
    required_columns: Optional[List[str]] = None,
) -> Tuple[RawData, RawDataInfo]:
    """Standardize data to consistent format.

    Args:
        raw_data: Input data
        raw_data_info: Input metadata
        technique_hint: Override technique detection (string or list of strings)
        custom_mapping: Additional column name mappings
        required_columns: List of columns that must be present

    Returns:
        Tuple of (RawData, RawDataInfo) with standardized data
    """
    # Extract dataset from RawData
    if isinstance(raw_data.data, xr.Dataset):
        # Determine techniques
        techniques = technique_hint or raw_data_info.get("technique") or ["unknown"]
        if isinstance(techniques, str):
            techniques = [techniques]

        # Standardize data
        standardizer = DataStandardizer(dataset=raw_data.data, techniques=techniques, instrument=raw_data_info.instrument)
        standardizer.standardize(custom_mapping)

        if required_columns:
            standardizer.ensure_required_columns(required_columns)

        standardized_data = RawData(data=standardizer.get_dataset())
    elif isinstance(raw_data.data, xr.DataTree):
        # Determine global techniques
        global_techniques = technique_hint or raw_data_info.get("technique") or ["unknown"]
        if isinstance(global_techniques, str):
            global_techniques = [global_techniques]

        def _standardize_node(ds: xr.Dataset) -> xr.Dataset:
            if not ds.data_vars:
                return ds

            # Use node-specific techniques if available, otherwise use global
            node_techniques = ds.attrs.get("technique", global_techniques)
            if isinstance(node_techniques, str):
                node_techniques = [node_techniques]

            s = DataStandardizer(dataset=ds, techniques=node_techniques, instrument=raw_data_info.instrument)
            s.standardize(custom_mapping)
            if required_columns:
                s.ensure_required_columns(required_columns)

            standardized_ds = s.get_dataset()
            # Sanitize names for DataTree compatibility (no '/' allowed)
            rename_dict = {str(var): str(var).replace("/", "_") for var in standardized_ds.data_vars if "/" in str(var)}
            if rename_dict:
                standardized_ds = standardized_ds.rename(rename_dict)

            return standardized_ds

        standardized_tree = raw_data.data.map_over_datasets(_standardize_node)
        standardized_data = RawData(data=standardized_tree)
        techniques = global_techniques
    else:
        raise ValueError("RawData must contain an xarray.Dataset or DataTree for standardization")

    # Create standardized info
    info = raw_data_info.copy()
    info.technique = techniques

    return standardized_data, info


__all__ = [
    "DataStandardizer",
    "standardize_names",
]
