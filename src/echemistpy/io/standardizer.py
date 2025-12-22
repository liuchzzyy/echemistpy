"""Data standardization utilities for echemistpy.

This module handles the conversion of raw measurement data into standardized
formats with consistent column names and units across different instruments
and techniques.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple

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

    # Standard column name mappings for different techniques
    STANDARD_MAPPINGS: ClassVar[dict[str, dict[str, str]]] = {
        "electrochemistry": {
            # Time variants
            "time": "Time/s",
            "Time": "Time/s",
            "TIME": "Time/s",
            "t": "Time/s",
            "time/s": "Time/s",
            "time_s": "Time/s",
            "Time_s": "Time/s",
            "test_time": "Time/s",
            # Cycle variants
            "Ns": "Cycle_number",
            "ns": "Cycle_number",
            "cycle": "Cycle_number",
            "Cycle": "Cycle_number",
            "cycle number": "Cycle_number",
            "cycle_number": "Cycle_number",
            "step": "Step_number",
            "step_number": "Step_number",
            "record": "Record_number",
            "record_number": "Record_number",
            # Potential/Voltage variants (working electrode)
            "potential": "Ewe/V",
            "Potential": "Ewe/V",
            "E": "Ewe/V",
            "Ewe": "Ewe/V",
            "Ewe/V": "Ewe/V",
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
            # Power variants
            "power": "P/W",
            "Power": "P/W",
            "P": "P/W",
            "P/W": "P/W",
            # EIS variants
            "freq/Hz": "freq/Hz",
            "Re(Z)/Ohm": "rez_ohm",
            "-Im(Z)/Ohm": "imz_ohm",
            "|Z|/Ohm": "z_ohm",
            "Phase(Z)/deg": "phasez_deg",
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

    # Standard metadata key mappings
    METADATA_MAPPINGS: ClassVar[dict[str, str]] = {
        "Acquisition started on": "start_time",
        "Acquisition ended on": "end_time",
        "Instrument": "instrument",
        "Operator": "operator",
        "Sample": "sample_name",
        "test_name": "sample_name",
        "Technique": "technique",
        "File": "filename",
        "Comments": "remarks",
        "Mass of active material": "active_material_mass",
        "Characteristic mass": "active_material_mass",
    }

    def __init__(self, dataset: xr.Dataset, techniques: list[str] | str = "unknown", **kwargs):
        """Initialize with a dataset and technique(s)."""
        if isinstance(techniques, str):
            techniques = [techniques]
        super().__init__(dataset=dataset.copy(deep=True), techniques=[t.lower() for t in techniques], **kwargs)

    def standardize_column_names(self, custom_mapping: Optional[Dict[str, str]] = None) -> "DataStandardizer":
        """Standardize column names based on techniques and custom mappings."""
        # Build aggregate mapping from all techniques
        mapping = {}
        for tech in self.techniques:
            # Map specific techniques to general categories if needed
            tech_category = tech
            if tech in {"cv", "gcd", "eis", "ca", "cp", "lsjv", "echem", "ec", "peis", "gpcl"}:
                tech_category = "electrochemistry"

            if tech_category in self.STANDARD_MAPPINGS:
                mapping.update(self.STANDARD_MAPPINGS[tech_category])

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

    def standardize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Standardize metadata keys based on standard mappings."""
        standardized = {}
        for key, value in metadata.items():
            new_key = self.METADATA_MAPPINGS.get(key, key)
            standardized[new_key] = value
        return standardized

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
            elif "capacity" in var_name.lower() or var_name.startswith("Q"):
                if "/uAh" in var_name:
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

    def validate_data_format(self) -> Dict[str, Any]:
        """Validate that data follows echemistpy format conventions."""
        issues = {"warnings": [], "errors": []}

        # Check record/row dimension
        if "record" not in self.dataset.coords and "row" not in self.dataset.coords:
            issues["errors"].append("Missing 'record' or 'row' coordinate dimension")

        # Check for technique-specific required columns
        technique_requirements = {
            "cv": ["Time/s", "Ewe/V", "Current/mA"],
            "gcd": ["Time/s", "Ewe/V", "Current/mA"],
            "eis": ["freq/Hz", "rez_ohm", "imz_ohm"],
            "peis": ["freq/Hz", "rez_ohm", "imz_ohm"],
            "xrd": ["2theta/deg", "intensity/counts"],
            "xps": ["BE/eV", "intensity/cps"],
            "tga": ["T/°C", "weight/%"],
        }

        for tech in self.techniques:
            if tech in technique_requirements:
                required = technique_requirements[tech]
                missing = [col for col in required if col not in self.dataset.data_vars]
                if missing:
                    issues["warnings"].append(f"Missing recommended columns for {tech}: {missing}")

        return issues

    def get_dataset(self) -> xr.Dataset:
        """Return the standardized dataset."""
        return self.dataset


def detect_technique(dataset: xr.Dataset) -> list[str]:
    """Auto-detect technique based on column names and data patterns."""
    columns = list(dataset.data_vars.keys())
    columns_lower = [str(col).lower() for col in columns]
    techniques = []

    # Check for electrochemistry patterns
    has_time = any("time" in col for col in columns_lower)
    has_potential = any(any(pot in col for pot in ["potential", "voltage", "ewe", " e ", " v "]) for col in columns_lower)
    has_current = any(any(curr in col for curr in ["current", " i ", "ma", "amp"]) for col in columns_lower)

    if has_time and has_potential and has_current:
        techniques.append("electrochemistry")

    # Check for EIS patterns
    has_frequency = any("freq" in col for col in columns_lower)
    has_impedance = any(any(imp in col for imp in ["z", "impedance", "re_z", "im_z"]) for col in columns_lower)
    if has_frequency and has_impedance:
        techniques.append("eis")

    # Check for XRD patterns
    has_angle = any(any(ang in col for ang in ["2theta", "angle", "theta"]) for col in columns_lower)
    has_intensity = any("intensity" in col or "counts" in col for col in columns_lower)
    if has_angle and has_intensity:
        techniques.append("xrd")

    # Check for XPS patterns
    has_be = any("be" in col or "binding" in col or "energy" in col for col in columns_lower)
    if has_be and has_intensity:
        techniques.append("xps")

    # Check for TGA patterns
    has_temp = any("temp" in col or " t " in col for col in columns_lower)
    has_weight = any("weight" in col or "mass" in col for col in columns_lower)
    if has_temp and has_weight:
        techniques.append("tga")

    # Default fallback
    if not techniques:
        return ["unknown"]
    return techniques


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
        dataset = raw_data.data
    else:
        raise ValueError("RawData must contain an xarray.Dataset for standardization")

    # Determine techniques
    techniques = technique_hint or raw_data_info.get("technique") or detect_technique(dataset)
    if isinstance(techniques, str):
        techniques = detect_technique(dataset) if techniques in {"Unknown", "unknown", "Table"} else [techniques]

    # Standardize data
    standardizer = DataStandardizer(dataset, techniques)
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

    # Standardize metadata
    raw_meta = raw_data_info.others
    standardized_meta = standardizer.standardize_metadata(raw_meta)

    # Create RawDataInfo with standardized fields
    info = RawDataInfo(
        technique=techniques,
        sample_name=standardized_meta.get("sample_name", "Unknown"),
        instrument=standardized_meta.get("instrument"),
        operator=standardized_meta.get("operator"),
        start_time=standardized_meta.get("start_time"),
        active_material_mass=standardized_meta.get("active_material_mass"),
        others=standardized_meta,
    )

    # Create RawData
    standardized_data = RawData(data=standardized_dataset)

    return standardized_data, info


__all__ = [
    "DataStandardizer",
    "detect_technique",
    "standardize_names",
]
