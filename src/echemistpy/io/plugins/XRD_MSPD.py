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
    DEFAULT_TECHNIQUE: ClassVar[List[str]] = ["xrd", "operando"]
    INSTRUMENT_NAME: ClassVar[str] = "ALBA_MSPD"

    # --- Loader Metadata ---
    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "mspd"

    # --- Traitlets ---
    filepath = Unicode()
    wave_number = Unicode(None, allow_none=True)
    sample_name = Unicode(None, allow_none=True)
    start_time = Unicode(None, allow_none=True)
    operator = Unicode(None, allow_none=True)
    active_material_mass = Unicode(None, allow_none=True)
    technique = TList(Unicode(), default_value=DEFAULT_TECHNIQUE)
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

        # Extract metadata
        start_time = self._extract_start_time(metadata)
        wave_val = self._extract_wave_number(metadata)

        raw_info = RawDataInfo(
            sample_name=self.sample_name or path.stem,
            start_time=start_time,
            technique=list(self.technique),
            instrument=self.instrument,
            operator=self.operator,
            active_material_mass=self.active_material_mass,
            wave_number=wave_val,
            others={**metadata, "file_path": str(path)},
        )

        return RawData(data=ds), raw_info

    def _extract_start_time(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract and format start time from metadata."""
        if self.start_time:
            return self.start_time

        start_time_str = metadata.get("Date")
        if start_time_str:
            try:
                dt = self.parse_date(start_time_str)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return start_time_str
        return None

    def _extract_wave_number(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract wave number from metadata or traits."""
        if self.wave_number:
            return self.wave_number
        wave = metadata.get("Wave")
        return str(wave) if wave is not None else None

    def _load_directory(self, path: Path) -> Tuple[RawData, RawDataInfo]:
        """Load all MSPD .xye files in a directory, merging files in the same subfolder."""
        xye_files = sorted(path.rglob("*.xye"))
        if not xye_files:
            raise FileNotFoundError(f"No .xye files found in {path}")

        # Group files by parent directory
        groups: Dict[Path, List[Path]] = {}
        for f in xye_files:
            groups.setdefault(f.parent, []).append(f)

        tree = xr.DataTree(name=path.name)
        all_infos: List[RawDataInfo] = []

        for parent, files in groups.items():
            try:
                merged_ds, node_infos = self._process_directory_group(files)
                if merged_ds is None:
                    continue

                # Determine relative path for the tree
                node_path = "/" if parent == path else "/".join(parent.relative_to(path).parts)

                # Add to tree
                target_node = tree if node_path == "/" else tree.create_node(node_path)
                target_node.dataset = merged_ds

                # Merge and store metadata
                node_info = self._merge_infos(node_infos, parent)
                target_node.attrs.update(node_info.to_dict())
                all_infos.append(node_info)

            except Exception as e:
                logger.error("Failed to load/merge files in %s: %s", parent, e)

        if not tree.children and not tree.has_data:
            raise RuntimeError(f"Failed to load any valid .xye files from {path}")

        # Merge all infos for the root
        root_info = self._merge_infos(all_infos, path)
        tree.attrs.update(root_info.to_dict())
        return RawData(data=tree), root_info

    def _process_directory_group(self, files: List[Path]) -> Tuple[Optional[xr.Dataset], List[RawDataInfo]]:
        """Process a group of files in the same directory."""
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

    def _merge_infos(self, infos: List[RawDataInfo], root_path: Path) -> RawDataInfo:
        """Merge multiple RawDataInfo objects into one."""
        if not infos:
            return RawDataInfo()

        base = infos[0]
        all_techs = {t for info in infos for t in info.technique}
        all_files = []

        for info in infos:
            files = info.get("merged_files")
            if files:
                all_files.extend(files)
            else:
                fp = info.get("file_path")
                if fp:
                    all_files.append(Path(fp).name)

        total_files = sum(info.get("n_files", 1) for info in infos)

        return RawDataInfo(
            sample_name=self.sample_name or root_path.name,
            start_time=self.start_time or base.start_time,
            operator=self.operator or base.operator,
            technique=list(all_techs),
            instrument=self.instrument or base.instrument,
            active_material_mass=self.active_material_mass or base.active_material_mass,
            wave_number=self.wave_number or base.wave_number,
            others={
                "n_files": total_files,
                "structure": "DataTree",
                "merged_files": all_files,
            },
        )

    def _read_single_xye(self, filepath: Path) -> Tuple[xr.Dataset, Dict[str, Any]]:
        """Read a single .xye file and return an xarray Dataset and metadata."""
        metadata = self._parse_header(filepath)

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
        ds = xr.Dataset(
            {
                "intensity": (("2theta",), df["intensity"].values),
                "intensity_error": (("2theta",), df["intensity_error"].values),
            },
            coords={"2theta": df["2theta"].values},
        )

        # Add units and long names
        ds.intensity.attrs.update({"units": "counts", "long_name": "Intensity"})
        ds.intensity_error.attrs.update({"units": "counts", "long_name": "Intensity Error"})
        ds["2theta"].attrs.update({"units": "degree", "long_name": "2-Theta"})

        # Calculate d-spacing if Wave is available
        wave_to_use = self._get_wave_to_use(metadata)
        if wave_to_use is not None:
            d_spacing = self.calculate_d_spacing(ds["2theta"].values, wave_to_use)
            ds = ds.assign_coords(d_spacing=(("2theta",), d_spacing))
            ds.d_spacing.attrs.update({"units": "Å", "long_name": "d-spacing"})

        return ds, metadata

    @staticmethod
    def _parse_header(filepath: Path) -> Dict[str, Any]:
        """Parse header for metadata."""
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
        """Determine the wavelength to use for d-spacing calculation."""
        if self.wave_number:
            try:
                return float(self.wave_number)
            except ValueError:
                logger.warning("Invalid wave_number trait: %s", self.wave_number)
        return metadata.get("Wave")

    @staticmethod
    def calculate_d_spacing(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
        """Calculate d-spacing from 2theta and wavelength using Bragg's Law."""
        theta_rad = np.deg2rad(two_theta / 2.0)
        return wavelength / (2.0 * np.sin(theta_rad))

    @classmethod
    def parse_date(cls, date_str: str) -> datetime:
        """Parse MSPD date string format 'YYYY-MM-DD_HH:MM:SS' to datetime object."""
        if not date_str:
            raise ValueError("Empty date string")
        return datetime.strptime(date_str, cls.DATE_FORMAT)
