"""Generic file loading and standardization for scientific measurements.

This module provides:
1. Format detection and plugin-based loading into RawMeasurement objects
2. Data standardization using technique-specific mappings
3. Unified API for loading and processing measurement data
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from .structures import (
    RawData,
    RawMetadata,
    RawMeasurement,
    Measurement,
    MeasurementInfo,
)

# Type alias for loader plugins
Loader = Callable[[Path, Any], RawMeasurement]


# ============================================================================
# Helper Functions
# ============================================================================


def _create_raw_measurement(
    dataset: xr.Dataset,
    path: Path,
    technique: str = "Unknown",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> RawMeasurement:
    """Wrap a dataset into a RawMeasurement with metadata.
    
    Args:
        dataset: xarray Dataset containing the data
        path: Path to the source file
        technique: Measurement technique
        extra_meta: Additional metadata to include
        
    Returns:
        RawMeasurement object with data and metadata
    """
    meta_dict = {
        "technique": technique,
        "source_file": str(path),
        "original_filename": path.name,
        "extension": path.suffix,
    }
    if extra_meta:
        meta_dict.update(extra_meta)

    metadata = RawMetadata(meta=meta_dict)
    raw_data = RawData(data=dataset)

    return RawMeasurement(data=raw_data, metadata=metadata)


def _structured_array_to_dataset(array: np.ndarray) -> xr.Dataset:
    """Convert structured numpy array to xarray Dataset."""
    if array.size == 0:
        return xr.Dataset()
    if array.ndim == 0:
        array = array.reshape(1)
    if array.dtype.names is None:
        raise ValueError("Tabular text files must include a header row.")
    row_dim = np.arange(array.shape[0])
    data_vars = {name: ("row", array[name]) for name in array.dtype.names}
    return xr.Dataset(data_vars=data_vars, coords={"row": row_dim})


# ============================================================================
# Format-Specific Loader Plugins
# ============================================================================


def _load_delimited(path: Path, *, delimiter: str, **kwargs: Any) -> RawMeasurement:
    """Load delimited text files (CSV, TSV, etc.)."""
    # Read header lines to capture potential metadata
    header_lines = []
    with open(path, "r", errors="ignore") as f:
        for _ in range(10):  # Read first 10 lines to check for comments
            line = f.readline()
            if line.strip().startswith("#") or line.strip().startswith("%"):
                header_lines.append(line.strip())
            else:
                break

    array = np.genfromtxt(
        path, delimiter=delimiter, names=True, dtype=None, encoding=None, **kwargs
    )
    dataset = _structured_array_to_dataset(array)

    extra_meta = {"header_comments": header_lines} if header_lines else {}
    return _create_raw_measurement(dataset, path, technique="Table", extra_meta=extra_meta)


def _load_lanhe_ccs(path: Path, **kwargs: Any) -> RawMeasurement:
    """Load LANHE .ccs binary files using LanheReader plugin."""
    tag_filter = kwargs.pop("tag_filter", None)
    channel_filter = kwargs.pop("channel_filter", None)
    
    from echemistpy.utils.external.echem.lanhe_reader import LanheReader

    reader = LanheReader(path)
    dataset = reader.to_dataset(tag_filter=tag_filter, channel_filter=channel_filter)

    # Extract metadata from reader if available
    extra_meta = {}
    if hasattr(reader, "header"):
        extra_meta["lanhe_header"] = reader.header

    return _create_raw_measurement(dataset, path, technique="Echem/Lanhe", extra_meta=extra_meta)


def _load_netcdf(path: Path, **kwargs: Any) -> RawMeasurement:
    """Load NetCDF files."""
    dataset = xr.open_dataset(path, **kwargs)
    # Extract global attributes as metadata
    extra_meta = dict(dataset.attrs)
    return _create_raw_measurement(dataset, path, technique="NetCDF", extra_meta=extra_meta)


def _load_biologic(path: Path, **kwargs: Any) -> RawMeasurement:
    """Load BioLogic .mpr or .mpt files using BiologicMPTReader plugin."""
    from echemistpy.utils.external.echem.biologic_reader import BiologicMPTReader

    reader = BiologicMPTReader()
    measurement = reader.read(path, **kwargs)

    # BiologicMPTReader already returns a RawMeasurement with complete metadata
    return measurement


def _load_excel(path: Path, **kwargs: Any) -> RawMeasurement:
    """Load Excel files using pandas backend."""

    # Read Excel file
    df = pd.read_excel(path, **kwargs)

    # Convert to xarray Dataset
    data_vars = {}
    for col in df.columns:
        data_vars[str(col)] = ("row", df[col].values)

    dataset = xr.Dataset(data_vars=data_vars, coords={"row": np.arange(len(df))})
    return _create_raw_measurement(dataset, path, technique="Excel")


def _load_hdf5(path: Path, **kwargs: Any) -> RawMeasurement:
    """Load HDF5 files using xarray backend."""
    try:
        dataset = xr.open_dataset(path, engine="h5netcdf", **kwargs)
    except Exception:
        # Fallback to netcdf4 engine
        dataset = xr.open_dataset(path, engine="netcdf4", **kwargs)

    extra_meta = dict(dataset.attrs)
    return _create_raw_measurement(dataset, path, technique="HDF5", extra_meta=extra_meta)


# ============================================================================
# Plugin Registry
# ============================================================================

_LOADER_MAP: Dict[str, Loader] = {
    "csv": lambda path, **kwargs: _load_delimited(path, delimiter=",", **kwargs),
    "tsv": lambda path, **kwargs: _load_delimited(path, delimiter="\t", **kwargs),
    "ccs": _load_lanhe_ccs,
    "nc": _load_netcdf,
    "nc4": _load_netcdf,
    "netcdf": _load_netcdf,
    "mpr": _load_biologic,
    "mpt": _load_biologic,
    "xlsx": _load_excel,
    "xls": _load_excel,
    "xlsm": _load_excel,
    "xlsb": _load_excel,
    "h5": _load_hdf5,
    "hdf5": _load_hdf5,
    "hdf": _load_hdf5,
}

# Alias for backwards compatibility
_LOADERS = _LOADER_MAP


# ============================================================================
# Unified Loader Interface
# ============================================================================


def _load(path: Path, fmt: Optional[str] = None, **kwargs: Any) -> RawMeasurement:
    """Unified loader function that dispatches to format-specific plugins.
    
    Args:
        path: Path to the data file
        fmt: Optional format override (file extension without dot)
        **kwargs: Additional arguments passed to the specific loader plugin
        
    Returns:
        RawMeasurement containing loaded data and metadata
        
    Raises:
        ValueError: If file format is not supported
    """
    extension = (fmt or path.suffix.lstrip(".")).lower()
    
    try:
        loader = _LOADER_MAP[extension]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported file extension '{extension}'. "
            f"Supported formats: {', '.join(sorted(_LOADER_MAP.keys()))}"
        ) from exc
    
    return loader(path, **kwargs)


def load_table(
    path: str | Path,
    *,
    fmt: Optional[str] = None,
    storage_options: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> RawMeasurement:
    """Load a tabular file into a RawMeasurement.

    Args:
        path: Path to the data file
        fmt: Optional format override (e.g., 'csv', 'tsv')
        storage_options: Not used yet, kept for API stability
        **kwargs: Additional arguments passed to the specific loader

    Returns:
        RawMeasurement containing the loaded data and metadata

    Examples:
        >>> from pathlib import Path
        >>> import tempfile
        >>> # Create a simple CSV file
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        ...     _ = f.write("voltage,current\\n3.1,0.5\\n3.4,0.4\\n")
        ...     temp_path = f.name
        >>> measurement = load_table(temp_path)
        >>> measurement.data.data["voltage"].values.tolist()
        [3.1, 3.4]
        >>> measurement.metadata.meta["technique"]
        'Table'
    """
    del storage_options  # not used yet but kept for API stability

    path = Path(path)
    return _load(path, fmt=fmt, **kwargs)


def load_data_file(path: str | Path, **kwargs: Any) -> RawMeasurement:
    """Universal data file loader with automatic format detection.

    This function attempts to load various scientific data formats by
    detecting the file extension and dispatching to the appropriate
    loader plugin.

    Args:
        path: Path to the data file
        **kwargs: Additional arguments passed to the specific loader

    Returns:
        RawMeasurement containing the loaded data and metadata

    Raises:
        ValueError: If no suitable loader can handle the file

    Examples:
        >>> # Load various file types automatically
        >>> measurement1 = load_data_file("data.csv")  # CSV file
        >>> measurement2 = load_data_file("data.xlsx") # Excel file
        >>> measurement3 = load_data_file("data.mpr")  # BioLogic file
        >>> measurement4 = load_data_file("data.ccs")  # LANHE file
    """
    path = Path(path)
    return _load(path, **kwargs)


def register_loader(extension: str, loader: Loader) -> None:
    """Register a custom loader plugin for a file extension.
    
    Args:
        extension: File extension (without dot, e.g., 'xyz')
        loader: Loader function that takes (Path, **kwargs) and returns RawMeasurement
        
    Example:
        >>> def my_custom_loader(path: Path, **kwargs) -> RawMeasurement:
        ...     # Custom loading logic
        ...     dataset = ...  # Create xarray Dataset
        ...     return _create_raw_measurement(dataset, path, technique="Custom")
        >>> register_loader("xyz", my_custom_loader)
    """
    _LOADER_MAP[extension.lower()] = loader


# ============================================================================
# Data Standardization
# ============================================================================


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

    def standardize_column_names(
        self, custom_mapping: Optional[Dict[str, str]] = None
    ) -> "DataStandardizer":
        """Standardize column names based on technique and custom mappings."""
        # Map specific techniques to general categories
        technique_category = self.technique
        if self.technique in ["cv", "gcd", "eis", "ca", "cp", "lsjv", "echem", "ec"]:
            technique_category = "electrochemistry"

        # Get standard mapping for technique
        if technique_category in self.STANDARD_MAPPINGS:
            mapping = self.STANDARD_MAPPINGS[technique_category].copy()
        else:
            mapping = {}

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
            elif (
                "voltage" in var_name.lower()
                or "potential" in var_name.lower()
                or var_name.startswith("E")
            ):
                if "/mV" in var_name:
                    # Convert mV to V
                    self.dataset[var_name] = var_data / 1000
                    new_name = var_name.replace("/mV", "/V")
                    self.dataset = self.dataset.rename({var_name: new_name})

        return self

    def ensure_required_columns(
        self, required_columns: List[str]
    ) -> "DataStandardizer":
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
                issues["warnings"].append(
                    f"Missing recommended columns for {self.technique}: {missing}"
                )

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
    has_potential = any(
        any(pot in col for pot in ["potential", "voltage", "ewe", " e ", " v "])
        for col in columns_lower
    )
    has_current = any(
        any(curr in col for curr in ["current", " i ", "ma", "amp"])
        for col in columns_lower
    )

    if has_time and has_potential and has_current:
        return "electrochemistry"

    # Check for EIS patterns
    has_frequency = any("freq" in col for col in columns_lower)
    has_impedance = any(
        any(imp in col for imp in ["z", "impedance", "re_z", "im_z"])
        for col in columns_lower
    )
    if has_frequency and has_impedance:
        return "eis"

    # Check for XRD patterns
    has_angle = any(
        any(ang in col for ang in ["2theta", "angle", "theta"])
        for col in columns_lower
    )
    has_intensity = any(
        "intensity" in col or "counts" in col for col in columns_lower
    )
    if has_angle and has_intensity:
        return "xrd"

    # Check for XPS patterns
    has_be = any(
        "be" in col or "binding" in col or "energy" in col for col in columns_lower
    )
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
    raw_measurement: RawMeasurement,
    technique_hint: Optional[str] = None,
    custom_mapping: Optional[Dict[str, str]] = None,
    required_columns: Optional[List[str]] = None,
) -> Tuple[Measurement, MeasurementInfo]:
    """Standardize a RawMeasurement into Measurement and MeasurementInfo.

    Args:
        raw_measurement: Input raw measurement
        technique_hint: Override technique detection
        custom_mapping: Additional column name mappings
        required_columns: List of columns that must be present

    Returns:
        Tuple of (Measurement, MeasurementInfo)

    Example:
        >>> raw = load_data_file("data.csv")
        >>> measurement, info = standardize_measurement(raw)
        >>> print(info.technique)
        'electrochemistry'
        >>> print(measurement.data.data_vars)
        Data variables:
            Time/s       (row) float64 ...
            Ewe/V        (row) float64 ...
            Current/mA   (row) float64 ...
    """
    # Extract dataset from RawData
    if isinstance(raw_measurement.data.data, xr.Dataset):
        dataset = raw_measurement.data.data
    else:
        raise ValueError(
            "RawData must contain an xarray.Dataset for standardization"
        )

    # Determine technique
    technique = (
        technique_hint
        or raw_measurement.metadata.meta.get("technique")
        or detect_technique(dataset)
    )
    if technique in ["Unknown", "unknown", "Table"]:
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
    raw_meta = raw_measurement.metadata.meta

    info = MeasurementInfo(
        technique=technique,
        sample_name=raw_meta.get("original_filename", "Unknown").split(".")[0],
        instrument=raw_meta.get("instrument"),
        operator=raw_meta.get("operator"),
        extras=raw_meta,  # Keep all raw metadata in extras
    )

    return Measurement(data=standardized_dataset), info


# ============================================================================
# Utility Functions
# ============================================================================


def get_file_info(path: str | Path) -> Dict[str, Any]:
    """Get basic information about a data file without fully loading it.

    Args:
        path: Path to the data file

    Returns:
        Dictionary with file information including size, format, columns, etc.
    """
    path = Path(path)
    info = {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "extension": path.suffix.lower(),
        "exists": path.exists(),
        "supported": path.suffix.lower().lstrip(".") in _LOADER_MAP,
    }

    if not path.exists():
        return info

    # Try to get column information for supported formats
    try:
        if info["extension"] in [".csv", ".tsv"]:
            # Read just the header
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                delimiter = "," if info["extension"] == ".csv" else "\t"
                columns = first_line.split(delimiter)
                info["columns"] = [col.strip() for col in columns]
                info["n_columns"] = len(columns)

    except Exception:
        # If we can't analyze, just note it
        info["analysis_error"] = "Could not analyze file structure"

    return info


def list_supported_formats() -> Dict[str, str]:
    """Return a dictionary of supported file formats and their descriptions."""
    return {
        "csv": "Comma-separated values",
        "tsv": "Tab-separated values",
        "ccs": "LANHE proprietary cycler files",
        "nc/nc4/netcdf": "NetCDF format",
        "mpr/mpt": "BioLogic EC-Lab files",
        "xlsx/xls": "Excel spreadsheet (requires pandas)",
        "h5/hdf5/hdf": "HDF5 format",
    }


__all__ = [
    # Loading functions
    "load_data_file",
    "load_table",
    "register_loader",
    # Standardization
    "standardize_measurement",
    "DataStandardizer",
    "detect_technique",
    # Utilities
    "get_file_info",
    "list_supported_formats",
    # Type alias
    "Loader",
]
