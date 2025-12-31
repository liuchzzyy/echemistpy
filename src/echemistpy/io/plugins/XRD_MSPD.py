# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""XRD Data Reader for MSPD .xye files with metadata extraction using traitlets."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from traitlets import HasTraits, Unicode
from traitlets import List as TList

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class MSPDReader(HasTraits):
    """Reader for MSPD XRD .xye files.

    Supports reading a single .xye file or a directory containing multiple .xye files.
    When reading a directory, files are organized into an xarray.DataTree.
    """

    # --- Constants ---
    DATE_FORMAT: ClassVar[str] = "%Y-%m-%d_%H:%M:%S"
    DEFAULT_TECHNIQUE_SINGLE: ClassVar[List[str]] = ["xrd", "in_situ"]
    DEFAULT_TECHNIQUE_DIR: ClassVar[List[str]] = ["xrd", "operando"]
    INSTRUMENT_NAME: ClassVar[str] = "ALBA_MSPD"

    # --- Loader Metadata ---
    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "alba_mspd"

    # --- Traitlets ---
    filepath = Unicode()
    wave_number = Unicode(None, allow_none=True)
    sample_name = Unicode(None, allow_none=True)
    start_time = Unicode(None, allow_none=True)
    operator = Unicode(None, allow_none=True)
    active_material_mass = Unicode(None, allow_none=True)
    technique = TList(Unicode(), default_value=None, allow_none=True)
    # instrument traitlet removed to avoid conflict with ClassVar
    # if needed, it can be a property or just use the ClassVar

    def __init__(self, filepath: Optional[Union[str, Path]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self, **_kwargs: Any) -> Tuple[RawData, RawDataInfo]:
        """Load MSPD .xye file(s) and return RawData and RawDataInfo.

        Returns:
            Tuple of (RawData, RawDataInfo)
        """
        if not self.filepath:
            raise ValueError("filepath must be set before calling load()")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return self._load_single_file(path)
        if path.is_dir():
            return self._load_directory(path)

        raise ValueError(f"Path is neither a file nor a directory: {path}")

    def _load_single_file(self, path: Path) -> Tuple[RawData, RawDataInfo]:
        """Internal method to load a single MSPD .xye file."""
        ds, metadata = self._read_single_xye(path)

        # Clean and enhance metadata
        metadata["file_path"] = str(path)
        cleaned_metadata = self._clean_metadata(metadata, path)

        # Extract wave number
        wave_val = self._extract_wave_number(cleaned_metadata)

        # 单一文件使用 in_situ 技术类型
        technique = list(self.technique) if self.technique else self.DEFAULT_TECHNIQUE_SINGLE

        raw_info = RawDataInfo(
            sample_name=self.sample_name or cleaned_metadata.get("sample_name", path.stem),
            start_time=self.start_time or cleaned_metadata.get("start_time"),
            technique=technique,
            instrument=self.instrument,
            operator=self.operator or cleaned_metadata.get("operator"),
            active_material_mass=self.active_material_mass or cleaned_metadata.get("active_material_mass"),
            wave_number=wave_val,
            others=cleaned_metadata,
        )

        return RawData(data=ds), raw_info

    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any], filepath: Path) -> Dict[str, Any]:
        """Clean and structure metadata from raw header data.

        Args:
            metadata: Raw metadata dictionary from file header
            filepath: Path to the file being processed

        Returns:
            Cleaned metadata dictionary with standardized keys
        """
        cleaned: Dict[str, Any] = {}

        # Extract start time
        if "Date" in metadata:
            try:
                dt = MSPDReader.parse_date(metadata["Date"])
                cleaned["start_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                cleaned["start_time"] = metadata["Date"]

        # Extract wavelength
        if "Wave" in metadata:
            cleaned["wavelength"] = metadata["Wave"]

        # Store sample name from filename
        cleaned["sample_name"] = filepath.stem
        cleaned["file_path"] = str(filepath)

        return cleaned

    def _extract_wave_number(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract wave number from metadata or traits.

        Returns wavelength as a string for RawDataInfo.wave_number field.
        """
        if self.wave_number:
            return self.wave_number
        wavelength = metadata.get("wavelength")
        return str(wavelength) if wavelength is not None else None

    def _load_directory(self, path: Path) -> Tuple[RawData, RawDataInfo]:
        """Load all MSPD .xye files in a directory into a DataTree."""
        xye_files = sorted(path.rglob("*.xye"))
        if not xye_files:
            raise FileNotFoundError(f"No .xye files found in {path}")

        # Group files by parent directory
        groups: Dict[Path, List[Path]] = {}
        for f in xye_files:
            groups.setdefault(f.parent, []).append(f)

        tree_dict: Dict[str, xr.Dataset] = {}
        all_infos: List[RawDataInfo] = []

        for parent, files in groups.items():
            try:
                merged_ds, node_infos = self._process_directory_group(files)
                if merged_ds is None:
                    continue

                # Determine relative path for the tree
                if parent == path:
                    node_path = "/"
                else:
                    rel_path = parent.relative_to(path)
                    node_path = "/" + "/".join(rel_path.parts)

                tree_dict[node_path] = merged_ds

                # Merge and store metadata for this node
                node_info = self._merge_node_infos(node_infos, parent)
                merged_ds.attrs.update(node_info.to_dict())
                all_infos.append(node_info)

            except Exception as e:
                logger.error("Failed to load/merge files in %s: %s", parent, e)

        if not tree_dict:
            raise RuntimeError(f"Failed to load any valid .xye files from {path}")

        # Create DataTree from dictionary
        tree = xr.DataTree.from_dict(tree_dict, name=path.name)

        # Merge all infos for the root
        root_info = self._merge_infos(all_infos, path)
        tree.attrs.update(root_info.to_dict())

        return RawData(data=tree), root_info

    def _process_directory_group(self, files: List[Path]) -> Tuple[Optional[xr.Dataset], List[RawDataInfo]]:
        """Process a group of files in the same directory.

        Loads all files and concatenates them along the 'record' dimension.
        Adds time coordinates (systime and time_s) based on file timestamps.

        Args:
            files: List of .xye file paths in the same directory

        Returns:
            Tuple of (merged_dataset, list_of_infos) or (None, []) if all files failed
        """
        datasets = []
        infos = []
        for f in files:
            try:
                raw_data, raw_info = self._load_single_file(f)
                datasets.append(raw_data.data)
                infos.append(raw_info)
            except Exception as e:
                logger.warning("Skipping file %s due to error: %s", f, e)

        if not datasets:
            return None, []

        # Merge datasets along 'record' dimension
        merged_ds = xr.concat(datasets, dim="record")

        # Extract and convert times
        systimes = pd.to_datetime([info.start_time for info in infos])
        rel_times = (systimes - systimes[0]).total_seconds() if not systimes.isnull().all() else [np.nan] * len(infos)

        # Add coordinates
        merged_ds = merged_ds.assign_coords(
            record=np.arange(len(datasets)),
            filename=("record", [f.name for f in files]),
            systime=("record", systimes),
            time_s=("record", rel_times),
        )

        # Add metadata to coordinates
        merged_ds.time_s.attrs.update({"units": "s", "long_name": "Relative Time"})
        merged_ds.systime.attrs.update({"long_name": "System Time"})

        return merged_ds, infos

    def _merge_node_infos(self, infos: List[RawDataInfo], parent_path: Path) -> RawDataInfo:
        """Merge RawDataInfo objects for a single directory node.

        This is used when merging files within the same directory.
        """
        if not infos:
            return RawDataInfo()

        # Collect sample names (one per file)
        sample_names = [info.sample_name for info in infos if info.sample_name]

        # Collect unique operators and start times
        operators = sorted({info.operator for info in infos if info.operator})
        start_times = sorted({info.start_time for info in infos if info.start_time})
        masses = sorted({info.active_material_mass for info in infos if info.active_material_mass})
        wave_numbers = sorted({info.wave_number for info in infos if info.wave_number})

        # Use directory-specific technique
        technique = list(self.technique) if self.technique else self.DEFAULT_TECHNIQUE_DIR

        # Build others dict
        others = {
            "n_files": len(infos),
            "sample_names": sample_names,  # Store all sample names
            "filenames": [Path(info.get("file_path", "")).name for info in infos if info.get("file_path")],
        }

        # Add lists if multiple unique values
        if len(operators) > 1:
            others["all_operators"] = operators
        if len(masses) > 1:
            others["all_active_material_masses"] = masses
        if len(wave_numbers) > 1:
            others["all_wave_numbers"] = wave_numbers

        return RawDataInfo(
            sample_name=self.sample_name or parent_path.name,
            technique=technique,
            instrument=self.instrument,
            operator=self.operator or (operators[0] if len(operators) == 1 else None),
            start_time=self.start_time or (start_times[0] if len(start_times) == 1 else None),
            active_material_mass=self.active_material_mass or (masses[0] if len(masses) == 1 else None),
            wave_number=self.wave_number or (wave_numbers[0] if len(wave_numbers) == 1 else None),
            others=others,
        )

    def _merge_infos(self, infos: List[RawDataInfo], root_path: Path) -> RawDataInfo:
        """Merge multiple RawDataInfo objects from different directories.

        This is used when combining information from multiple subdirectories
        into the root RawDataInfo.
        """
        if not infos:
            return RawDataInfo()

        # Collect all sample names from all subdirectories
        all_sample_names = []
        for info in infos:
            # Each info might have multiple sample_names in its 'others'
            if "sample_names" in info.others:
                all_sample_names.extend(info.others["sample_names"])
            elif info.sample_name:
                all_sample_names.append(info.sample_name)

        # Collect unique values
        operators = sorted({info.operator for info in infos if info.operator})
        start_times = sorted({info.start_time for info in infos if info.start_time})
        masses = sorted({info.active_material_mass for info in infos if info.active_material_mass})
        wave_numbers = sorted({info.wave_number for info in infos if info.wave_number})

        # Use directory-specific technique
        technique = list(self.technique) if self.technique else self.DEFAULT_TECHNIQUE_DIR

        # Calculate total files
        total_files = sum(info.others.get("n_files", 1) for info in infos)

        # Determine folder name
        folder_name = root_path.resolve().name if root_path.name in (".", "..", "") else root_path.name

        # Build combined others dict
        combined_others = {
            "n_files": total_files,
            "sample_names": all_sample_names,
        }

        # Add lists if multiple unique values
        if len(operators) > 1:
            combined_others["all_operators"] = operators
        if len(masses) > 1:
            combined_others["all_active_material_masses"] = masses
        if len(wave_numbers) > 1:
            combined_others["all_wave_numbers"] = wave_numbers

        return RawDataInfo(
            sample_name=self.sample_name or folder_name,
            technique=technique,
            instrument=self.instrument,
            operator=self.operator or (operators[0] if len(operators) == 1 else None),
            start_time=self.start_time or (start_times[0] if len(start_times) == 1 else None),
            active_material_mass=self.active_material_mass or (masses[0] if len(masses) == 1 else None),
            wave_number=self.wave_number or (wave_numbers[0] if len(wave_numbers) == 1 else None),
            others=combined_others,
        )

    def _read_single_xye(self, filepath: Path) -> Tuple[xr.Dataset, Dict[str, Any]]:
        """Read a single .xye file and return an xarray Dataset and metadata.

        Args:
            filepath: Path to the .xye file

        Returns:
            Tuple of (Dataset, metadata_dict)
        """
        # Parse header metadata
        metadata = self._parse_header(filepath)

        # Read data columns
        try:
            df = pd.read_csv(
                filepath,
                comment="#",
                sep=r"\s+",
                names=["2theta", "intensity", "intensity_error"],
                engine="python",
            )
        except Exception as e:
            logger.error("Error reading %s: %s", filepath, e)
            raise

        # Create xarray Dataset
        ds = self._create_dataset(df, metadata)

        return ds, metadata

    def _create_dataset(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> xr.Dataset:
        """Create a standardized xarray.Dataset from DataFrame and metadata.

        Args:
            df: DataFrame with columns ['2theta', 'intensity', 'intensity_error']
            metadata: Metadata dictionary from file header

        Returns:
            xarray.Dataset with data variables and coordinates
        """
        # Create Dataset with data variables
        ds = xr.Dataset(
            {
                "intensity": (("2theta",), df["intensity"].values),
                "intensity_error": (("2theta",), df["intensity_error"].values),
            },
            coords={"2theta": df["2theta"].values},
        )

        # Apply standard attributes
        self._apply_standard_attrs(ds)

        # Calculate and add d-spacing if wavelength is available
        wave_to_use = self._get_wave_to_use(metadata)
        if wave_to_use is not None:
            d_spacing = self.calculate_d_spacing(ds["2theta"].values, wave_to_use)
            ds = ds.assign_coords(d_spacing=(("2theta",), d_spacing))
            ds.d_spacing.attrs.update({"units": "Å", "long_name": "d-spacing"})

        return ds

    @staticmethod
    def _apply_standard_attrs(ds: xr.Dataset) -> None:
        """Apply standard units and long names to Dataset variables.

        Args:
            ds: xarray.Dataset to modify in-place
        """
        attr_map = {
            "intensity": {"units": "counts", "long_name": "Intensity"},
            "intensity_error": {"units": "counts", "long_name": "Intensity Error"},
            "2theta": {"units": "degree", "long_name": "2-Theta"},
        }

        for var, attrs in attr_map.items():
            if var in ds:
                ds[var].attrs.update(attrs)
            if var in ds.coords:
                ds.coords[var].attrs.update(attrs)

    @staticmethod
    def _parse_header(filepath: Path) -> Dict[str, Any]:
        """Parse header lines from .xye file for metadata.

        Extracts information from lines starting with '#'.
        Currently extracts:
        - Wave: X-ray wavelength in Angstroms
        - Date: Acquisition timestamp

        Args:
            filepath: Path to the .xye file

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                # Extract Wave
                if "Wave =" in line and (match := re.search(r"Wave\s*=\s*([\d\.]+)", line)):
                    metadata["Wave"] = float(match.group(1))
                # Extract Date
                if "Date =" in line and (match := re.search(r"Date\s*=\s*([\d\-_:]+)", line)):
                    metadata["Date"] = match.group(1)
        return metadata

    def _get_wave_to_use(self, metadata: Dict[str, Any]) -> Optional[float]:
        """Determine the wavelength to use for d-spacing calculation.

        Priority:
        1. User-provided wave_number traitlet
        2. Wave value from file metadata

        Args:
            metadata: Metadata dictionary from file

        Returns:
            Wavelength in Angstroms, or None if not available
        """
        if self.wave_number:
            try:
                return float(self.wave_number)
            except ValueError:
                logger.warning("Invalid wave_number trait: %s", self.wave_number)
        return metadata.get("Wave")

    @staticmethod
    def calculate_d_spacing(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
        """Calculate d-spacing from 2theta and wavelength using Bragg's Law.

        Bragg's Law: nλ = 2d sinθ
        For n=1: d = λ / (2 sinθ)

        Args:
            two_theta: Array of 2θ values in degrees
            wavelength: X-ray wavelength in Angstroms

        Returns:
            Array of d-spacing values in Angstroms
        """
        theta_rad = np.deg2rad(two_theta / 2.0)
        return wavelength / (2.0 * np.sin(theta_rad))

    @classmethod
    def parse_date(cls, date_str: str) -> datetime:
        """Parse MSPD date string format 'YYYY-MM-DD_HH:MM:SS' to datetime object.

        Args:
            date_str: Date string in MSPD format

        Returns:
            datetime object

        Raises:
            ValueError: If date string is empty or cannot be parsed
        """
        if not date_str:
            raise ValueError("Empty date string")
        return datetime.strptime(date_str, cls.DATE_FORMAT)
