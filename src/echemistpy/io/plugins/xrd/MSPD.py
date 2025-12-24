# -*- coding: utf-8 -*-
"""XRD Data Reader for MSPD .xye files with metadata extraction using traitlets."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from traitlets import HasTraits, List, Unicode

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class MSPDReader(HasTraits):
    """Reader for MSPD XRD .xye files.

    Supports reading a single .xye file or a directory containing multiple .xye files.
    When reading a directory, files are organized into an xarray.DataTree.
    """

    filepath = Unicode()
    wave_number = Unicode(None, allow_none=True)
    sample_name = Unicode(None, allow_none=True)
    start_time = Unicode(None, allow_none=True)
    operator = Unicode(None, allow_none=True)
    active_material_mass = Unicode(None, allow_none=True)
    technique = List(Unicode(), default_value=["xrd", "operando"])
    instrument = Unicode("ALBA_MSPD", allow_none=True)

    def __init__(self, filepath: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self) -> tuple[RawData, RawDataInfo]:
        """Load MSPD .xye file(s) and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath not set")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return self._load_single_file(path)
        elif path.is_dir():
            return self._load_directory(path)
        else:
            raise ValueError(f"Path is neither a file nor a directory: {path}")

    def _load_single_file(self, path: Path) -> tuple[RawData, RawDataInfo]:
        """Internal method to load a single MSPD .xye file."""
        ds, metadata = self._read_single_xye(path)

        # Create RawDataInfo
        start_time = self.start_time
        if start_time is None:
            start_time_str = metadata.get("Date")
            if start_time_str:
                try:
                    dt = self.parse_date(start_time_str)
                    start_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    start_time = start_time_str

        wave_val = self.wave_number
        if wave_val is None:
            wave = metadata.get("Wave")
            wave_val = str(wave) if wave is not None else None

        raw_info = RawDataInfo(
            sample_name=self.sample_name or path.stem,
            start_time=start_time,
            technique=self.technique,
            instrument=self.instrument,
            operator=self.operator,
            active_material_mass=self.active_material_mass,
            wave_number=wave_val,
            others={**metadata, "file_path": str(path)},
        )

        return RawData(data=ds), raw_info

    def _load_directory(self, path: Path) -> tuple[RawData, RawDataInfo]:
        """Load all MSPD .xye files in a directory, merging files in the same subfolder."""
        xye_files = sorted(path.rglob("*.xye"))
        if not xye_files:
            raise FileNotFoundError(f"No .xye files found in {path}")

        # Group files by parent directory
        groups = {}
        for f in xye_files:
            parent = f.parent
            if parent not in groups:
                groups[parent] = []
            groups[parent].append(f)

        tree = xr.DataTree(name=path.name)
        all_infos = []

        for parent, files in groups.items():
            try:
                # Load and merge files in this directory
                datasets = []
                infos = []
                for f in files:
                    raw_data, raw_info = self._load_single_file(f)
                    datasets.append(raw_data.data)
                    infos.append(raw_info)

                if not datasets:
                    continue

                # Merge datasets along 'record' dimension
                merged_ds = xr.concat(datasets, dim="record")

                # Extract and convert times
                systimes = pd.to_datetime([info.start_time for info in infos])

                # Calculate relative time (timedelta)
                if not systimes.isnull().all():
                    rel_times = systimes - systimes[0]
                else:
                    rel_times = pd.to_timedelta([None] * len(infos))

                # Add record, filenames, systime and time_s as coordinates
                merged_ds = merged_ds.assign_coords(
                    record=np.arange(len(datasets)),
                    filename=("record", [f.name for f in files]),
                    systime=("record", systimes),
                )
                merged_ds.coords["time_s"] = ("record", rel_times)
                merged_ds.coords["time_s"].attrs["units"] = "s"
                merged_ds.coords["time_s"].attrs["long_name"] = "Relative Time"

                # Determine relative path for the tree
                if parent == path:
                    node_path = "/"
                else:
                    rel_path = parent.relative_to(path)
                    node_path = "/".join(rel_path.parts)

                if node_path == "/" or node_path == "":
                    tree.dataset = merged_ds
                    node_info = self._merge_infos(infos, parent)
                    tree.attrs.update(node_info.to_dict())
                    all_infos.append(node_info)
                else:
                    tree[node_path] = merged_ds
                    node_info = self._merge_infos(infos, parent)
                    tree[node_path].attrs.update(node_info.to_dict())
                    all_infos.append(node_info)

            except Exception as e:
                logger.warning("Failed to load/merge files in %s: %s", parent, e)

        if not tree.children and not tree.has_data:
            raise RuntimeError(f"Failed to load any .xye files from {path}")

        # Merge all infos for the root
        root_info = self._merge_infos(all_infos, path)
        tree.attrs.update(root_info.to_dict())
        return RawData(data=tree), root_info

    def _merge_infos(self, infos: list[RawDataInfo], root_path: Path) -> RawDataInfo:
        """Merge multiple RawDataInfo objects into one."""
        if not infos:
            return RawDataInfo()

        base = infos[0]
        all_techs = set()
        all_files = []
        for info in infos:
            for t in info.technique:
                all_techs.add(t)

            # Collect filenames
            files = info.get("merged_files")
            if files:
                all_files.extend(files)
            else:
                # Single file case
                fp = info.get("file_path")
                if fp:
                    all_files.append(Path(fp).name)

        # Calculate total files
        total_files = 0
        for info in infos:
            n = info.get("n_files")
            if n is not None:
                total_files += n
            else:
                # It's a single file info
                total_files += 1

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

    def _read_single_xye(self, filepath: Path) -> tuple[xr.Dataset, dict[str, Any]]:
        """Read a single .xye file and return an xarray Dataset and metadata."""
        metadata = {}

        # Parse header for metadata
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#"):
                    # Extract Wave
                    if "Wave =" in line:
                        match = re.search(r"Wave\s*=\s*([\d\.]+)", line)
                        if match:
                            metadata["Wave"] = float(match.group(1))
                    # Extract Date
                    if "Date =" in line:
                        match = re.search(r"Date\s*=\s*([\d\-_:]+)", line)
                        if match:
                            metadata["Date"] = match.group(1)
                else:
                    # Data starts
                    break

        # Read data using pandas
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

        # Calculate d-spacing if Wave is available
        wave_to_use = None
        if self.wave_number:
            try:
                wave_to_use = float(self.wave_number)
            except ValueError:
                logger.warning("Invalid wave_number trait: %s", self.wave_number)

        if wave_to_use is None and "Wave" in metadata:
            wave_to_use = metadata["Wave"]

        if wave_to_use is not None:
            ds = ds.assign_coords(
                d_spacing=(
                    ("2theta",),
                    self.calculate_d_spacing(ds["2theta"].values, wave_to_use),
                )
            )

        return ds, metadata

    @staticmethod
    def calculate_d_spacing(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
        """Calculate d-spacing from 2theta and wavelength using Bragg's Law."""
        theta_rad = np.deg2rad(two_theta / 2.0)
        return wavelength / (2.0 * np.sin(theta_rad))

    @staticmethod
    def parse_date(date_str: str) -> datetime:
        """Parse MSPD date string format 'YYYY-MM-DD_HH:MM:SS' to datetime object."""
        if not date_str:
            raise ValueError("Empty date string")
        return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")
