"""Data standardization utilities for echemistpy.

This module handles the conversion of raw measurement data into standardized
formats with consistent column names and units across different instruments
and techniques.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from echemistpy.io.structures import (
    Measurement,
    MeasurementInfo,
    RawData,
    RawDataInfo,
)


class DataStandardizer:
    """Standardize measurement data to echemistpy analysis format."""

    # Standard column name mappings for different techniques
    STANDARD_MAPPINGS = {
        "electrochemistry": {
            # Time variants
            "time": "Time/s",
            "Time": "Time/s",
            "TIME": "Time/s",
            "t": "Time/s",
            "time/s": "Time/s",
            "time_s": "Time/s",
            "Time_s": "Time/s",
            # Cycle variants
            "Ns": "Cycle_number",
            "ns": "Cycle_number",
            "cycle": "Cycle_number",
            "Cycle": "Cycle_number",
            "cycle number": "Cycle_number",
            # Potential/Voltage variants (working electrode)
            "potential": "Ewe/V",
            "Potential": "Ewe/V",
            "E": "Ewe/V",
            "Ewe": "Ewe/V",
            "Ewe/V": "Ewe/V",
            "potential_V": "Ewe/V",
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
            # Current variants
            "current": "Current/mA",
            "Current": "Current/mA",
            "I": "Current/mA",
            "i": "Current/mA",
            "current_mA": "Current/mA",
            "I_mA": "Current/mA",
            "<I>/mA": "Current/mA",
            "control/V/mA": "Current/mA",
            # Charge/Capacity variants
            "charge": "Q/mA.h",
            "Charge": "Q/mA.h",
            "Q": "Q/mA.h",
            "capacity": "Capacity/mAh",
            "Capacity": "Capacity/mAh",
            "(Q-Qo)/mA.h": "Capacity/mAh",
            # Power variants
            "power": "P/W",
            "Power": "P/W",
            "P": "P/W",
            "P/W": "P/W",
        },
        "xrd": {
            "2theta": "2theta/deg",
            "2Theta": "2theta/deg",
            "angle": "2theta/deg",
            "intensity": "intensity/counts",
            "Intensity": "intensity/counts",
            "counts": "intensity/counts",
            "Counts": "intensity/counts",
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

    def __init__(self, dataset: xr.Dataset, technique: str = "unknown"):
        """Initialize with a dataset and technique."""
        self.dataset = dataset.copy(deep=True)
        self.technique = technique.lower()

    def standardize_column_names(self, custom_mapping: Optional[Dict[str, str]] = None) -> "DataStandardizer":
        """Standardize column names based on technique and custom mappings."""
        # Map specific techniques to general categories
        technique_category = self.technique
        if self.technique in {"cv", "gcd", "eis", "ca", "cp", "lsjv", "echem", "ec"}:
            technique_category = "electrochemistry"

        # Get standard mapping for technique
        mapping = self.STANDARD_MAPPINGS[technique_category].copy() if technique_category in self.STANDARD_MAPPINGS else {}

        # Add custom mappings if provided
        if custom_mapping:
            mapping.update(custom_mapping)

        # Apply renaming
        rename_dict = {}
        for old_name in self.dataset.data_vars:
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
                elif "/µA" in var_name or "/uA" in var_name:
                    # Convert µA to mA
                    self.dataset[var_name] = var_data / 1000
                    new_name = var_name.replace("/µA", "/mA").replace("/uA", "/mA")
                    self.dataset = self.dataset.rename({var_name: new_name})

            # Handle voltage conversions
            elif ("voltage" in var_name.lower() or "potential" in var_name.lower() or var_name.startswith("E")) and "/mV" in var_name:
                # Convert mV to V
                self.dataset[var_name] = var_data / 1000
                new_name = var_name.replace("/mV", "/V")
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
            if "row" in self.dataset.coords:
                n_rows = len(self.dataset.coords["row"])
                for col in missing_cols:
                    self.dataset[col] = ("row", np.full(n_rows, np.nan))

        return self

    def validate_data_format(self) -> Dict[str, Any]:
        """Validate that data follows echemistpy format conventions."""
        issues = {"warnings": [], "errors": []}

        # Check row dimension
        if "row" not in self.dataset.coords:
            issues["errors"].append("Missing 'row' coordinate dimension")

        # Check for technique-specific required columns
        technique_requirements = {
            "cv": ["Time/s", "Ewe/V", "Current/mA"],
            "gcd": ["Time/s", "Ewe/V", "Current/mA"],
            "eis": ["freq/Hz", "Re_Z/Ohm", "Im_Z/Ohm"],
            "xrd": ["2theta/deg", "intensity/counts"],
            "xps": ["BE/eV", "intensity/cps"],
            "tga": ["T/°C", "weight/%"],
        }

        if self.technique in technique_requirements:
            required = technique_requirements[self.technique]
            missing = [col for col in required if col not in self.dataset.data_vars]
            if missing:
                issues["warnings"].append(f"Missing recommended columns for {self.technique}: {missing}")

        return issues

    def get_dataset(self) -> xr.Dataset:
        """Return the standardized dataset."""
        return self.dataset


def detect_technique(dataset: xr.Dataset) -> str:
    """Auto-detect measurement technique based on column names and data patterns."""
    columns = list(dataset.data_vars.keys())
    columns_lower = [col.lower() for col in columns]

    # Check for electrochemistry patterns
    has_time = any("time" in col for col in columns_lower)
    has_potential = any(any(pot in col for pot in ["potential", "voltage", "ewe", " e ", " v "]) for col in columns_lower)
    has_current = any(any(curr in col for curr in ["current", " i ", "ma", "amp"]) for col in columns_lower)

    if has_time and has_potential and has_current:
        return "electrochemistry"

    # Check for EIS patterns
    has_frequency = any("freq" in col for col in columns_lower)
    has_impedance = any(any(imp in col for imp in ["z", "impedance", "re_z", "im_z"]) for col in columns_lower)
    if has_frequency and has_impedance:
        return "eis"

    # Check for XRD patterns
    has_angle = any(any(ang in col for ang in ["2theta", "angle", "theta"]) for col in columns_lower)
    has_intensity = any("intensity" in col or "counts" in col for col in columns_lower)
    if has_angle and has_intensity:
        return "xrd"

    # Check for XPS patterns
    has_be = any("be" in col or "binding" in col or "energy" in col for col in columns_lower)
    if has_be and has_intensity:
        return "xps"

    # Check for TGA patterns
    has_temp = any("temp" in col or " t " in col for col in columns_lower)
    has_weight = any("weight" in col or "mass" in col for col in columns_lower)
    if has_temp and has_weight:
        return "tga"

    # Default fallback
    return "unknown"


def standardize_measurement(
    raw_data: RawData,
    raw_data_info: RawDataInfo,
    technique_hint: Optional[str] = None,
    custom_mapping: Optional[Dict[str, str]] = None,
    required_columns: Optional[List[str]] = None,
) -> Tuple[Measurement, MeasurementInfo]:
    """Standardize raw data to measurement format.

    Args:
        raw_data: Input raw data
        raw_data_info: Input raw data metadata
        technique_hint: Override technique detection
        custom_mapping: Additional column name mappings
        required_columns: List of columns that must be present

    Returns:
        Tuple of (Measurement, MeasurementInfo) with standardized data
    """
    # Extract dataset from RawData
    if isinstance(raw_data.data, xr.Dataset):
        dataset = raw_data.data
    else:
        raise ValueError("RawData must contain an xarray.Dataset for standardization")

    # Determine technique
    technique = technique_hint or raw_data_info.get("technique") or detect_technique(dataset)
    if technique in {"Unknown", "unknown", "Table"}:
        technique = detect_technique(dataset)

    # Standardize data
    standardizer = DataStandardizer(dataset, technique)
    standardizer.standardize_column_names(custom_mapping)
    standardizer.standardize_units()

    if required_columns:
        standardizer.ensure_required_columns(required_columns)

    # Validate
    issues = standardizer.validate_data_format()
    if issues["warnings"]:
        for warning in issues["warnings"]:
            warnings.warn(warning, stacklevel=2)

    standardized_dataset = standardizer.get_dataset()

    # Create MeasurementInfo
    raw_meta = raw_data_info.meta

    info = MeasurementInfo(
        others={
            "technique": technique,
            "sample_name": raw_meta.get("filename", "Unknown"),
            "instrument": raw_meta.get("instrument"),
            "operator": raw_meta.get("operator"),
            **raw_meta,  # Keep all raw metadata in others
        }
    )

    # Create Measurement
    measurement = Measurement(data=standardized_dataset)

    return measurement, info


__all__ = [
    "DataStandardizer",
    "detect_technique",
    "standardize_measurement",
]
