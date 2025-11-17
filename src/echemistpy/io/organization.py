"""Data cleaning and organization functions for post-loading processing.

This module provides utilities to clean, validate, and organize measurement data
after it has been loaded by the loaders module but before analysis.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np

from .structures import Measurement


class DataCleaner:
    """Main class for cleaning and organizing measurement data."""

    def __init__(self, measurement: Measurement):
        """Initialize the data cleaner with a measurement object.

        Args:
            measurement: The measurement object to clean
        """
        self.measurement = measurement
        self.original_data = measurement.data.copy(deep=True)

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> "DataCleaner":
        """Remove duplicate rows based on specified columns.

        Args:
            subset: List of column names to consider for identifying duplicates.
                   If None, considers all columns.

        Returns:
            Self for method chaining
        """
        if subset is None:
            # Use all data variables for duplicate detection
            subset = list(self.measurement.data.data_vars.keys())

        # Create a combined array for duplicate detection
        arrays = [self.measurement.data[var].values for var in subset if var in self.measurement.data]
        if not arrays:
            warnings.warn("No valid columns found for duplicate removal", stacklevel=2)
            return self

        combined = np.column_stack(arrays)
        _, unique_indices = np.unique(combined, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)

        # Filter the dataset to keep only unique rows
        self.measurement.data = self.measurement.data.isel(row=unique_indices)

        return self

    def remove_outliers(self, columns: List[str], method: str = "iqr", factor: float = 1.5) -> "DataCleaner":
        """Remove outliers from specified columns.

        Args:
            columns: List of column names to check for outliers
            method: Method for outlier detection ('iqr', 'zscore', 'mad')
            factor: Factor for outlier threshold (1.5 for IQR, 3.0 for z-score)

        Returns:
            Self for method chaining
        """
        valid_mask = np.ones(len(self.measurement.data.row), dtype=bool)

        for col in columns:
            if col not in self.measurement.data:
                warnings.warn(f"Column '{col}' not found in data", stacklevel=2)
                continue

            values = self.measurement.data[col].values
            if not np.issubdtype(values.dtype, np.number):
                continue

            # Remove NaN values for outlier calculation
            valid_values = values[~np.isnan(values)]
            if len(valid_values) == 0:
                continue

            if method == "iqr":
                q1, q3 = np.percentile(valid_values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - factor * iqr
                upper_bound = q3 + factor * iqr
                col_mask = (values >= lower_bound) & (values <= upper_bound)

            elif method == "zscore":
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                z_scores = np.abs((values - mean_val) / std_val)
                col_mask = z_scores <= factor

            elif method == "mad":
                median_val = np.median(valid_values)
                mad_val = np.median(np.abs(valid_values - median_val))
                modified_z_scores = 0.6745 * (values - median_val) / mad_val
                col_mask = np.abs(modified_z_scores) <= factor

            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            # Handle NaN values - keep them unless they're outliers
            col_mask = col_mask | np.isnan(values)
            valid_mask = valid_mask & col_mask

        if np.sum(valid_mask) < len(valid_mask):
            self.measurement.data = self.measurement.data.isel(row=valid_mask)

        return self

    def fill_missing_values(self, columns: Optional[List[str]] = None, method: str = "interpolate", **kwargs) -> "DataCleaner":
        """Fill missing values in the dataset.

        Args:
            columns: List of columns to process. If None, processes all numeric columns
            method: Method for filling ('interpolate', 'forward', 'backward', 'mean', 'median')
            **kwargs: Additional arguments for the filling method

        Returns:
            Self for method chaining
        """
        if columns is None:
            # Auto-detect numeric columns
            columns = [var for var in self.measurement.data.data_vars if np.issubdtype(self.measurement.data[var].dtype, np.number)]

        for col in columns:
            if col not in self.measurement.data:
                warnings.warn(f"Column '{col}' not found in data", stacklevel=2)
                continue

            data_array = self.measurement.data[col]

            if method == "interpolate":
                filled = data_array.interpolate_na(dim="row", **kwargs)
            elif method == "forward":
                filled = data_array.ffill(dim="row", **kwargs)
            elif method == "backward":
                filled = data_array.bfill(dim="row", **kwargs)
            elif method == "mean":
                mean_val = data_array.mean(skipna=True)
                filled = data_array.fillna(mean_val)
            elif method == "median":
                median_val = data_array.median(skipna=True)
                filled = data_array.fillna(median_val)
            else:
                raise ValueError(f"Unknown filling method: {method}")

            self.measurement.data[col] = filled

        return self

    def normalize_column_names(self, name_mapping: Optional[Dict[str, str]] = None) -> "DataCleaner":
        """Standardize column names according to echemistpy conventions.

        Args:
            name_mapping: Optional custom mapping of old names to new names.
                         If None, applies standard echemistpy naming conventions.

        Returns:
            Self for method chaining
        """
        if name_mapping is None:
            # Standard naming conventions for common electrochemistry variables
            name_mapping = {
                # Time variants
                "time": "time/s",
                "Time": "time/s",
                "TIME": "time/s",
                "t": "time/s",
                "Time/s": "time/s",
                # Potential/Voltage variants
                "potential": "Ewe/V",
                "Potential": "Ewe/V",
                "voltage": "Ewe/V",
                "Voltage": "Ewe/V",
                "E": "Ewe/V",
                "V": "Ewe/V",
                "Ewe": "Ewe/V",
                # Current variants
                "current": "I/mA",
                "Current": "I/mA",
                "I": "I/mA",
                "i": "I/mA",
                # Charge variants
                "charge": "Q/mA.h",
                "Charge": "Q/mA.h",
                "Q": "Q/mA.h",
                "capacity": "Q/mA.h",
                # Power variants
                "power": "P/W",
                "Power": "P/W",
                "P": "P/W",
            }

        # Apply renaming only for columns that exist and have mappings
        rename_dict = {old_name: new_name for old_name, new_name in name_mapping.items() if old_name in self.measurement.data.data_vars}

        if rename_dict:
            self.measurement.data = self.measurement.data.rename(rename_dict)

        return self

    def validate_data_types(self, type_mapping: Optional[Dict[str, np.dtype]] = None) -> "DataCleaner":
        """Ensure data columns have appropriate data types.

        Args:
            type_mapping: Dictionary mapping column names to desired numpy dtypes.
                         If None, applies standard type conversions.

        Returns:
            Self for method chaining
        """
        if type_mapping is None:
            # Standard type mappings for electrochemistry data
            type_mapping = {
                "time/s": np.float64,
                "Ewe/V": np.float64,
                "I/mA": np.float64,
                "Q/mA.h": np.float64,
                "P/W": np.float64,
            }

        for col, target_dtype in type_mapping.items():
            if col in self.measurement.data.data_vars:
                try:
                    self.measurement.data[col] = self.measurement.data[col].astype(target_dtype)
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Could not convert column '{col}' to {target_dtype}: {e}", stacklevel=2)

        return self

    def sort_by_column(self, column: str, ascending: bool = True) -> "DataCleaner":
        """Sort the dataset by a specified column.

        Args:
            column: Column name to sort by
            ascending: If True, sort in ascending order

        Returns:
            Self for method chaining
        """
        if column not in self.measurement.data.data_vars:
            raise ValueError(f"Column '{column}' not found in data")

        sort_indices = np.argsort(self.measurement.data[column].values)
        if not ascending:
            sort_indices = sort_indices[::-1]

        self.measurement.data = self.measurement.data.isel(row=sort_indices)

        return self

    def filter_by_range(self, column: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> "DataCleaner":
        """Filter data to keep only rows within specified range for a column.

        Args:
            column: Column name to filter by
            min_val: Minimum value (inclusive). If None, no lower bound
            max_val: Maximum value (inclusive). If None, no upper bound

        Returns:
            Self for method chaining
        """
        if column not in self.measurement.data.data_vars:
            raise ValueError(f"Column '{column}' not found in data")

        values = self.measurement.data[column]
        mask = np.ones(len(values), dtype=bool)

        if min_val is not None:
            mask = mask & (values >= min_val)
        if max_val is not None:
            mask = mask & (values <= max_val)

        self.measurement.data = self.measurement.data.isel(row=mask)

        return self

    def apply_custom_function(self, func, columns: Optional[List[str]] = None) -> "DataCleaner":
        """Apply a custom function to specified columns.

        Args:
            func: Function to apply to each column
            columns: List of columns to apply function to. If None, applies to all numeric columns

        Returns:
            Self for method chaining
        """
        if columns is None:
            columns = [var for var in self.measurement.data.data_vars if np.issubdtype(self.measurement.data[var].dtype, np.number)]

        for col in columns:
            if col in self.measurement.data.data_vars:
                try:
                    self.measurement.data[col] = self.measurement.data[col].pipe(func)
                except Exception as e:
                    warnings.warn(f"Failed to apply function to column '{col}': {e}", stacklevel=2)

        return self

    def get_cleaned_measurement(self) -> Measurement:
        """Return the cleaned measurement object.

        Returns:
            The cleaned Measurement object
        """
        return self.measurement

    def get_cleaning_summary(self) -> Dict[str, any]:
        """Get a summary of the cleaning operations performed.

        Returns:
            Dictionary containing cleaning statistics
        """
        original_rows = len(self.original_data.row)
        current_rows = len(self.measurement.data.row)

        summary = {
            "original_rows": original_rows,
            "current_rows": current_rows,
            "rows_removed": original_rows - current_rows,
            "removal_percentage": ((original_rows - current_rows) / original_rows) * 100,
            "original_columns": list(self.original_data.data_vars.keys()),
            "current_columns": list(self.measurement.data.data_vars.keys()),
            "columns_renamed": set(self.original_data.data_vars.keys()) != set(self.measurement.data.data_vars.keys()),
        }

        return summary


def clean_measurement(measurement: Measurement, cleaning_steps: Optional[List[str]] = None, **kwargs) -> Measurement:
    """Convenience function to apply standard cleaning steps to a measurement.

    Args:
        measurement: The measurement object to clean
        cleaning_steps: List of cleaning steps to apply. If None, applies standard steps.
        **kwargs: Additional arguments for specific cleaning methods

    Returns:
        Cleaned measurement object

    Examples:
        >>> # Apply default cleaning
        >>> cleaned = clean_measurement(measurement)

        >>> # Apply specific cleaning steps
        >>> cleaned = clean_measurement(
        ...     measurement,
        ...     cleaning_steps=["normalize_names", "remove_duplicates", "sort_by_time"]
        ... )
    """
    if cleaning_steps is None:
        cleaning_steps = ["normalize_names", "validate_types", "remove_duplicates", "fill_missing", "sort_by_time"]

    cleaner = DataCleaner(measurement)

    for step in cleaning_steps:
        if step == "normalize_names":
            cleaner.normalize_column_names()
        elif step == "validate_types":
            cleaner.validate_data_types()
        elif step == "remove_duplicates":
            cleaner.remove_duplicates()
        elif step == "remove_outliers":
            outlier_columns = kwargs.get("outlier_columns", ["Ewe/V", "I/mA"])
            outlier_method = kwargs.get("outlier_method", "iqr")
            cleaner.remove_outliers(outlier_columns, method=outlier_method)
        elif step == "fill_missing":
            fill_method = kwargs.get("fill_method", "interpolate")
            cleaner.fill_missing_values(method=fill_method)
        elif step == "sort_by_time":
            time_col = kwargs.get("time_column", "time/s")
            if time_col in cleaner.measurement.data.data_vars:
                cleaner.sort_by_column(time_col)
        else:
            warnings.warn(f"Unknown cleaning step: {step}", stacklevel=2)

    return cleaner.get_cleaned_measurement()


def validate_measurement_integrity(measurement: Measurement) -> Dict[str, any]:
    """Validate the integrity of a measurement object.

    Args:
        measurement: The measurement object to validate

    Returns:
        Dictionary containing validation results
    """
    validation_results = {"is_valid": True, "errors": [], "warnings": [], "statistics": {}}

    # Check basic structure
    if measurement.data is None:
        validation_results["errors"].append("Data is None")
        validation_results["is_valid"] = False
        return validation_results

    # Check for empty dataset
    if len(measurement.data.data_vars) == 0:
        validation_results["errors"].append("No data variables found")
        validation_results["is_valid"] = False

    # Check for consistent row dimension
    row_lengths = []
    for var in measurement.data.data_vars:
        if "row" in measurement.data[var].dims:
            row_lengths.append(len(measurement.data[var]))

    if len(set(row_lengths)) > 1:
        validation_results["errors"].append("Inconsistent row lengths across variables")
        validation_results["is_valid"] = False

    # Check for missing values
    missing_counts = {}
    for var in measurement.data.data_vars:
        if np.issubdtype(measurement.data[var].dtype, np.number):
            missing_count = np.sum(np.isnan(measurement.data[var].values))
            if missing_count > 0:
                missing_counts[var] = missing_count

    if missing_counts:
        validation_results["warnings"].append(f"Missing values found: {missing_counts}")

    # Calculate basic statistics
    if row_lengths:
        validation_results["statistics"]["total_rows"] = row_lengths[0]
        validation_results["statistics"]["total_columns"] = len(measurement.data.data_vars)
        validation_results["statistics"]["missing_values"] = missing_counts

    return validation_results


class DataStandardizer:
    """Class for standardizing measurement data to echemistpy analysis format."""

    # Standard column name mappings for different techniques
    STANDARD_MAPPINGS = {
        "electrochemistry": {
            # Time variants
            "time": "time/s",
            "Time": "time/s",
            "TIME": "time/s",
            "t": "time/s",
            "Time/s": "time/s",
            "time_s": "time/s",
            "Time_s": "time/s",
            # Potential/Voltage variants
            "potential": "Ewe/V",
            "Potential": "Ewe/V",
            "voltage": "Ewe/V",
            "Voltage": "Ewe/V",
            "E": "Ewe/V",
            "V": "Ewe/V",
            "Ewe": "Ewe/V",
            "potential_V": "Ewe/V",
            "voltage_V": "Ewe/V",
            # Current variants
            "current": "I/mA",
            "Current": "I/mA",
            "I": "I/mA",
            "i": "I/mA",
            "current_mA": "I/mA",
            "I_mA": "I/mA",
            "<I>/mA": "I/mA",
            # Charge variants
            "charge": "Q/mA.h",
            "Charge": "Q/mA.h",
            "Q": "Q/mA.h",
            "capacity": "Q/mA.h",
            "Capacity": "Q/mA.h",
            # Power variants
            "power": "P/W",
            "Power": "P/W",
            "P": "P/W",
        },
        "xrd": {
            # XRD specific mappings
            "2theta": "2theta/deg",
            "2Theta": "2theta/deg",
            "angle": "2theta/deg",
            "intensity": "intensity/counts",
            "Intensity": "intensity/counts",
            "counts": "intensity/counts",
            "Counts": "intensity/counts",
        },
        "xps": {
            # XPS specific mappings
            "binding_energy": "BE/eV",
            "BE": "BE/eV",
            "energy": "BE/eV",
            "intensity": "intensity/cps",
            "Intensity": "intensity/cps",
            "counts": "intensity/cps",
            "cps": "intensity/cps",
        },
        "tga": {
            # TGA specific mappings
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

    def __init__(self, measurement: Measurement):
        """Initialize with a measurement object."""
        self.measurement = measurement
        self.technique = measurement.metadata.technique.lower()

    def standardize_column_names(self, custom_mapping: Optional[Dict[str, str]] = None) -> "DataStandardizer":
        """Standardize column names based on technique and custom mappings."""
        # Get standard mapping for technique
        if self.technique in self.STANDARD_MAPPINGS:
            mapping = self.STANDARD_MAPPINGS[self.technique].copy()
        else:
            mapping = {}

        # Add custom mappings if provided
        if custom_mapping:
            mapping.update(custom_mapping)

        # Apply renaming
        rename_dict = {}
        for old_name in self.measurement.data.data_vars:
            if old_name in mapping:
                rename_dict[old_name] = mapping[old_name]

        if rename_dict:
            self.measurement.data = self.measurement.data.rename(rename_dict)

        return self

    def standardize_units(self) -> "DataStandardizer":
        """Convert units to standard echemistpy conventions."""
        for var_name in list(self.measurement.data.data_vars.keys()):
            var_data = self.measurement.data[var_name]

            # Handle time conversions
            if "time" in var_name.lower():
                if "min" in var_name:
                    # Convert minutes to seconds
                    self.measurement.data[var_name] = var_data * 60
                    new_name = var_name.replace("min", "s")
                    self.measurement.data = self.measurement.data.rename({var_name: new_name})
                elif "h" in var_name and "mA.h" not in var_name:
                    # Convert hours to seconds
                    self.measurement.data[var_name] = var_data * 3600
                    new_name = var_name.replace("h", "s")
                    self.measurement.data = self.measurement.data.rename({var_name: new_name})

            # Handle current conversions
            elif "current" in var_name.lower() or var_name.startswith("I"):
                if "/A" in var_name or "_A" in var_name:
                    # Convert A to mA
                    self.measurement.data[var_name] = var_data * 1000
                    new_name = var_name.replace("/A", "/mA").replace("_A", "_mA")
                    self.measurement.data = self.measurement.data.rename({var_name: new_name})
                elif "/µA" in var_name or "/uA" in var_name:
                    # Convert µA to mA
                    self.measurement.data[var_name] = var_data / 1000
                    new_name = var_name.replace("/µA", "/mA").replace("/uA", "/mA")
                    self.measurement.data = self.measurement.data.rename({var_name: new_name})

            # Handle voltage conversions
            elif "voltage" in var_name.lower() or "potential" in var_name.lower() or var_name.startswith("E"):
                if "/mV" in var_name:
                    # Convert mV to V
                    self.measurement.data[var_name] = var_data / 1000
                    new_name = var_name.replace("/mV", "/V")
                    self.measurement.data = self.measurement.data.rename({var_name: new_name})

        return self

    def ensure_required_columns(self, required_columns: List[str]) -> "DataStandardizer":
        """Ensure that required columns exist, creating placeholders if needed."""
        missing_cols = []
        for col in required_columns:
            if col not in self.measurement.data.data_vars:
                missing_cols.append(col)

        if missing_cols:
            warnings.warn(f"Missing required columns: {missing_cols}. Creating placeholders.")
            # Create placeholder columns with NaN values
            n_rows = len(self.measurement.data.coords["row"])
            for col in missing_cols:
                self.measurement.data[col] = ("row", np.full(n_rows, np.nan))

        return self

    def validate_data_format(self) -> Dict[str, any]:
        """Validate that data follows echemistpy format conventions."""
        issues = {"warnings": [], "errors": []}

        # Check row dimension
        if "row" not in self.measurement.data.coords:
            issues["errors"].append("Missing 'row' coordinate dimension")

        # Check for technique-specific required columns
        technique_requirements = {
            "cv": ["time/s", "Ewe/V", "I/mA"],
            "gcd": ["time/s", "Ewe/V", "I/mA"],
            "eis": ["freq/Hz", "Re_Z/Ohm", "Im_Z/Ohm"],
            "xrd": ["2theta/deg", "intensity/counts"],
            "xps": ["BE/eV", "intensity/cps"],
            "tga": ["T/°C", "weight/%"],
        }

        if self.technique in technique_requirements:
            required = technique_requirements[self.technique]
            missing = [col for col in required if col not in self.measurement.data.data_vars]
            if missing:
                issues["warnings"].append(f"Missing recommended columns for {self.technique}: {missing}")

        # Check data types
        for var_name in self.measurement.data.data_vars:
            var_data = self.measurement.data[var_name]
            if not np.issubdtype(var_data.dtype, np.number):
                if var_data.dtype == object:
                    issues["warnings"].append(f"Column '{var_name}' has object dtype - may need type conversion")

        return issues

    def get_standardized_measurement(self) -> Measurement:
        """Return the standardized measurement object."""
        return self.measurement


def standardize_measurement(
    measurement: Measurement, technique_hint: Optional[str] = None, custom_mapping: Optional[Dict[str, str]] = None, required_columns: Optional[List[str]] = None
) -> Measurement:
    """Convenience function to standardize a measurement with default settings.

    Args:
        measurement: Input measurement to standardize
        technique_hint: Override technique detection (e.g., 'cv', 'gcd', 'xrd')
        custom_mapping: Additional column name mappings
        required_columns: List of columns that must be present

    Returns:
        Standardized measurement object
    """
    # Use technique hint if provided
    if technique_hint:
        measurement.metadata.technique = technique_hint

    standardizer = DataStandardizer(measurement)
    standardizer.standardize_column_names(custom_mapping)
    standardizer.standardize_units()

    if required_columns:
        standardizer.ensure_required_columns(required_columns)

    # Validate the result
    issues = standardizer.validate_data_format()
    if issues["warnings"]:
        for warning in issues["warnings"]:
            warnings.warn(warning, stacklevel=2)
    if issues["errors"]:
        raise ValueError(f"Data format errors: {issues['errors']}")

    return standardizer.get_standardized_measurement()


def detect_measurement_technique(measurement: Measurement) -> str:
    """Auto-detect measurement technique based on column names and data patterns.

    Args:
        measurement: Input measurement object

    Returns:
        Detected technique string (e.g., 'cv', 'gcd', 'eis', 'xrd', 'xps', 'tga')
    """
    columns = list(measurement.data.data_vars.keys())
    columns_lower = [col.lower() for col in columns]

    # Check for electrochemistry patterns
    has_time = any("time" in col for col in columns_lower)
    has_potential = any(any(pot in col for pot in ["potential", "voltage", "ewe", " e ", " v "]) for col in columns_lower)
    has_current = any(any(curr in col for curr in ["current", " i ", "ma", "amp"]) for col in columns_lower)

    if has_time and has_potential and has_current:
        # Distinguish between CV and GCD
        if len(measurement.data.coords["row"]) > 100:  # CV typically has more points
            return "cv"
        else:
            return "gcd"

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
