# -*- coding: utf-8 -*-
"""XRD Data Reader for NOTOs .xye files with metadata extraction using traitlets."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from traitlets import HasTraits, List, Unicode

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class NOTOsReader(HasTraits):
    """Reader for NOTOs XRD .xye files.

    Supports reading a single .xye file or a directory containing multiple .xye files.
    When reading a directory, files are merged along a 'record' dimension.
    """

    filepath = Unicode()
    wave_number = Unicode(None, allow_none=True)
    sample_name = Unicode(None, allow_none=True)
    start_time = Unicode(None, allow_none=True)
    operator = Unicode(None, allow_none=True)
    active_material_mass = Unicode(None, allow_none=True)
    technique = List(Unicode(), default_value=["xrd", "operando"])
    instrument = Unicode("NOTOs", allow_none=True)

    def __init__(self, filepath: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self) -> tuple[RawData, RawDataInfo]:
        """Load NOTOs .xye file(s) and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath not set")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        files = self._get_files(path)
        datasets = []
        all_metadata = []

        for f in files:
            ds, meta = self._read_single_xye(f)
            datasets.append(ds)
            all_metadata.append(meta)

        # Merge datasets
        if len(datasets) > 1:
            combined_ds = self._merge_datasets(datasets, all_metadata)
        else:
            combined_ds = datasets[0]
            # Add metadata as attributes for single file
            combined_ds.attrs.update(all_metadata[0])
            combined_ds.attrs["filename"] = files[0].name

        # Create RawData and RawDataInfo
        raw_data = RawData(data=combined_ds)
        raw_info = self._create_raw_info(path, files, all_metadata)

        return raw_data, raw_info

    @staticmethod
    def _get_files(path: Path) -> list[Path]:
        """Get list of .xye files from path."""
        if path.is_dir():
            files = sorted(path.glob("*.xye"))
            if not files:
                raise FileNotFoundError(f"No .xye files found in directory: {path}")
            return files

        if path.suffix.lower() != ".xye":
            logger.warning("File extension %s is not .xye", path.suffix)
        return [path]

    @staticmethod
    def _merge_datasets(datasets: list[xr.Dataset], all_metadata: list[dict[str, Any]]) -> xr.Dataset:
        """Merge multiple datasets along record dimension."""
        # Stack along 'record' dimension
        combined_ds = xr.concat(datasets, dim="record")

        # Parse dates to datetime objects and calculate relative time
        date_strs = [m.get("Date") for m in all_metadata]
        dates = []
        for d_str in date_strs:
            if isinstance(d_str, str):
                try:
                    dates.append(NOTOsReader.parse_date(d_str))
                except (ValueError, TypeError):
                    dates.append(None)
            else:
                dates.append(None)

        # Calculate relative time in seconds from the first scan
        rel_times = []
        start_dt = next((d for d in dates if d is not None), None)
        if start_dt:
            for d in dates:
                if d:
                    rel_times.append((d - start_dt).total_seconds())
                else:
                    rel_times.append(np.nan)
        else:
            rel_times = np.arange(len(datasets))

        return combined_ds.assign_coords(
            record=np.arange(1, len(datasets) + 1),
            datetime=("record", dates),
            time_s=("record", rel_times),
        )

    def _create_raw_info(self, path: Path, files: list[Path], all_metadata: list[dict[str, Any]]) -> RawDataInfo:
        """Create RawDataInfo object."""
        # Start time priority: self.start_time > file metadata
        start_time = self.start_time
        if start_time is None:
            start_time_str = all_metadata[0].get("Date")
            if start_time_str:
                try:
                    dt = self.parse_date(start_time_str)
                    start_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    start_time = start_time_str

        # Get wave number: priority to self.wave_number, then file metadata
        wave_val = self.wave_number
        if wave_val is None:
            wave = all_metadata[0].get("Wave")
            wave_val = str(wave) if wave is not None else None

        return RawDataInfo(
            sample_name=self.sample_name or (path.name if path.is_dir() else path.stem),
            start_time=start_time,
            technique=self.technique,
            instrument=self.instrument,
            operator=self.operator,
            active_material_mass=self.active_material_mass,
            wave_number=wave_val,
            others={
                "files": [f.name for f in files],
                "all_metadata": all_metadata,
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
        # .xye files typically have 3 columns: 2theta, intensity, error
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
        # We use '2theta' as the coordinate
        ds = xr.Dataset(
            {
                "intensity": (("2theta",), df["intensity"].values),
                "intensity_error": (("2theta",), df["intensity_error"].values),
            },
            coords={"2theta": df["2theta"].values},
        )

        # Calculate d-spacing if Wave is available
        # Priority: self.wave_number > metadata["Wave"]
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
        """Calculate d-spacing from 2theta and wavelength using Bragg's Law.

        Args:
            two_theta: 2theta values in degrees.
            wavelength: Wavelength in Angstroms.

        Returns:
            d-spacing values in the same units as wavelength.
        """
        # Convert 2theta to theta in radians
        theta_rad = np.deg2rad(two_theta / 2.0)
        # Bragg's Law: n*lambda = 2d*sin(theta) => d = lambda / (2*sin(theta))
        return wavelength / (2.0 * np.sin(theta_rad))

    @staticmethod
    def parse_date(date_str: str) -> datetime:
        """Parse NOTOs date string format 'YYYY-MM-DD_HH:MM:SS' to datetime object.

        Args:
            date_str: Date string from .xye file.

        Returns:
            datetime object.
        """
        if not date_str:
            raise ValueError("Empty date string")
        # Format: 2025-09-23_17:48:41
        return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")
