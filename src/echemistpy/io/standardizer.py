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
            "Systime": "systime",
            "SysTime": "systime",
            "systime": "systime",
            "abs_time": "systime",
            "Absolute Time": "systime",
            "DateTime": "systime",
            # Relative Time (Standardized to time_s)
            "time": "time_s",
            "Time": "time_s",
            "TIME": "time_s",
            "t": "time_s",
            "time/s": "time_s",
            "Time/s": "time_s",
            "time_s": "time_s",
            "Time_s": "time_s",
            "test_time": "time_s",
            "TestTime": "time_s",
            "test time": "time_s",
            "Test Time": "time_s",
            # Cycle variants
            "Ns": "cycle_number",
            "ns": "cycle_number",
            "cycle": "cycle_number",
            "Cycle": "cycle_number",
            "cycle number": "cycle_number",
            "cycle_number": "cycle_number",
            "Cycle Number": "cycle_number",
            "Cycle_Number": "cycle_number",
            "step": "step_number",
            "step_number": "step_number",
            "Step": "step_number",
            "Step Number": "step_number",
            "record": "record",
            "record_number": "record",
            "Record": "record",
            # Potential/Voltage variants (working electrode)
            "potential": "ewe_v",
            "Potential": "ewe_v",
            "E": "ewe_v",
            "Ewe": "ewe_v",
            "Ewe/V": "ewe_v",
            "Ece/V": "ece_v",
            "potential_V": "ewe_v",
            "voltage_v": "ewe_v",
            "Potential/V": "ewe_v",
            # Battery voltage variants (cell voltage)
            "voltage": "voltage_v",
            "Voltage": "voltage_v",
            "V": "voltage_v",
            "voltage_V": "voltage_v",
            "battery_voltage": "voltage_v",
            "Battery_Voltage": "voltage_v",
            "V_batt": "voltage_v",
            "Vbatt": "voltage_v",
            "cell_voltage": "voltage_v",
            "Cell_Voltage": "voltage_v",
            "voltage/V": "voltage_v",
            "Voltage/V": "voltage_v",
            # Current variants
            "current": "current_ma",
            "Current": "current_ma",
            "I": "current_ma",
            "i": "current_ma",
            "current_mA": "current_ma",
            "I_mA": "current_ma",
            "<I>/mA": "current_ma",
            "I/mA": "current_ma",
            "Current/mA": "current_ma",
            "control/V/mA": "current_ma",
            "current_ua": "current_ua",
            "current_ma": "current_ma",
            "current/uA": "current_ua",
            "current/mA": "current_ma",
            "Current/uA": "current_ua",
            # Charge/Capacity variants
            "charge": "q_mah",
            "Charge": "q_mah",
            "Q": "q_mah",
            "capacity": "capacity_mah",
            "Capacity": "capacity_mah",
            "(Q-Qo)/mA.h": "capacity_mah",
            "capacity_uah": "capacity_uah",
            "capacity_mah": "capacity_mah",
            "capacity/uAh": "capacity_uah",
            "capacity/mAh": "capacity_mah",
            "capacity/mA.h": "capacity_mah",
            "Capacity/mA.h": "capacity_mah",
            "Capacity/mAh": "capacity_mah",
            "capacity/uA.h": "capacity_uah",
            "Capacity/uA.h": "capacity_uah",
            "Capacity/uAh": "capacity_uah",
            "SpeCap/mAh/g": "specific_capacity_mah_g",
            "SpeCap_cal/mAh/g": "specific_capacity_cal_mah_g",
            "SpeCap_cal/mA.h/g": "specific_capacity_cal_mah_g",
            "Capacity_mA.h": "capacity_mah",
            "SpeCap_cal_mAh_g": "specific_capacity_cal_mah_g",
            # EIS variants
            "freq/Hz": "frequency_hz",
            "frequency": "frequency_hz",
            "Frequency": "frequency_hz",
            "Re(Z)/Ohm": "re_z_ohm",
            "Z'": "re_z_ohm",
            "Z_real": "re_z_ohm",
            "-Im(Z)/Ohm": "-im_z_ohm",
            "Z''": "-im_z_ohm",
            "Z_imag": "-im_z_ohm",
            "|Z|/Ohm": "z_mag_ohm",
            "Z_mag": "z_mag_ohm",
            "Phase(Z)/deg": "phase_deg",
            "phase": "phase_deg",
            "Phase": "phase_deg",
            "dQdV/uAh/V": "dqdv_uah_v",
            "dVdQ/V/uAh": "dvdq_v_uah",
        },
        "xrd": {
            "2theta": "2theta_degree",
            "2Theta": "2theta_degree",
            "angle": "2theta_degree",
            "intensity": "intensity",
            "Intensity": "intensity",
            "counts": "intensity",
            "Counts": "intensity",
            "d-spacing": "d-spacing_angstrom",
            "d_spacing": "d-spacing_angstrom",
        },
        "xps": {
            "binding_energy": "be_ev",
            "BE": "be_ev",
            "energy": "be_ev",
            "intensity": "intensity_cps",
            "Intensity": "intensity_cps",
            "counts": "intensity_cps",
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
        "xas": {
            "energy": "energy_eV",
            "Energy": "energy_eV",
            "energyc": "energy_eV",
            "energy_eV": "energy_eV",
            "absorption": "absorption_au",
            "Absorption": "absorption_au",
        },
        "txm": {
            "energy": "energy_eV",
            "x": "x_um",
            "y": "y_um",
            "transmission": "transmission",
            "optical_density": "optical_density",
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
                new_name = mapping[old_name]
                if new_name != old_name:
                    # Avoid conflicts if new_name already exists
                    if new_name in self.dataset:
                        # If the target name already exists (e.g. as a coordinate),
                        # we drop the old one to avoid redundancy.
                        if old_name in self.dataset:
                            self.dataset = self.dataset.drop_vars(old_name)
                        continue
                    rename_dict[old_name] = new_name

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

    def ensure_required_columns(self, required_columns: list[str]) -> "DataStandardizer":
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
    required_columns: Optional[list[str]] = None,
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
