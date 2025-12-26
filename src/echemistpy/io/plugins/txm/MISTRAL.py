# -*- coding: utf-8 -*-
"""TXM Data Reader for MISTRAL beamline HDF5 files."""

import logging
import re
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from traitlets import HasTraits, List, Unicode

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class MISTRALReader(HasTraits):
    """Reader for MISTRAL TXM .hdf5 files."""

    filepath = Unicode()
    sample_name = Unicode(None, allow_none=True)
    technique = List(Unicode(), default_value=["txm", "ex situ"])
    instrument = Unicode("ALBA_MISTRAL", allow_none=True)

    def __init__(self, filepath: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self, **_kwargs) -> tuple[RawData, RawDataInfo]:
        """Load MISTRAL HDF5 file(s) and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath not set")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return self._load_single_file(path)
        if path.is_dir():
            return self._load_directory(path)
        raise ValueError(f"Path is neither a file nor a directory: {path}")

    def _load_single_file(self, path: Path) -> tuple[RawData, RawDataInfo]:
        """Internal method to load a single MISTRAL HDF5 file."""
        with h5py.File(path, "r") as f:
            # Check if it's a SpecNormalized file
            if "SpecNormalized" not in f:
                raise ValueError(f"File {path} does not contain 'SpecNormalized' group.")

            group = f["SpecNormalized"]

            # Extract data
            data_cube = group["spectroscopy_normalized_aligned"][:]
            energy = group["energy"][:]

            # Optional data
            rotation_angle = group["rotation_angle"][:] if "rotation_angle" in group else None

            x_pixel_size = group["x_pixel_size"][0] if "x_pixel_size" in group else 1.0
            y_pixel_size = group["y_pixel_size"][0] if "y_pixel_size" in group else 1.0

            # Create coordinates
            x_coords = np.arange(data_cube.shape[2]) * x_pixel_size
            y_coords = np.arange(data_cube.shape[1]) * y_pixel_size

            # Create Dataset
            ds = xr.Dataset(
                data_vars={
                    "transmission": (["energy", "y", "x"], data_cube),
                    "optical_density": (["energy", "y", "x"], -np.log(data_cube.astype(np.float64))),
                },
                coords={
                    "energy": energy,
                    "y": y_coords,
                    "x": x_coords,
                },
                attrs={
                    "x_pixel_size": x_pixel_size,
                    "y_pixel_size": y_pixel_size,
                    "instrument": self.instrument,
                },
            )

            if rotation_angle is not None:
                ds["rotation_angle"] = (["energy"], rotation_angle)

            # Try to extract date from filename (e.g., 20230701)
            start_time = None
            date_match = re.search(r"(\d{8})", path.name)
            if date_match:
                try:
                    date_str = date_match.group(1)
                    start_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                except Exception as e:
                    logger.debug("Failed to parse date from filename %s: %s", path.name, e)

            # Metadata
            raw_info = RawDataInfo(
                sample_name=self.sample_name or path.stem,
                technique=self.technique,
                instrument=self.instrument,
                start_time=start_time,
                others={
                    "file_path": str(path),
                },
            )

            return RawData(data=ds), raw_info

    def _load_directory(self, path: Path) -> tuple[RawData, RawDataInfo]:
        """Load all MISTRAL HDF5 files in a directory into a DataTree."""
        all_files = sorted(path.rglob("*.hdf5*"))
        if not all_files:
            raise FileNotFoundError(f"No HDF5 files found in {path}")

        tree = xr.DataTree(name=path.name)
        all_infos = []

        for f in all_files:
            try:
                raw_data, raw_info = self._load_single_file(f)
                # Calculate relative path for the tree node
                rel_path = f.relative_to(path)
                # Use the relative path as node name, stripping extensions
                # e.g. folder1/file1.hdf5.hdf5 -> folder1/file1
                parts = list(rel_path.parts[:-1])
                file_stem = rel_path.name.split(".")[0]
                node_path = "/".join([*parts, file_stem])

                tree[node_path] = raw_data.data
                all_infos.append(raw_info)
            except Exception as e:
                logger.warning("Failed to load %s: %s", f, e)

        if not tree.children and not tree.has_data:
            raise RuntimeError(f"Failed to load any files from {path}")

        # Aggregate info
        first_info = all_infos[0] if all_infos else RawDataInfo()
        root_info = RawDataInfo(
            sample_name=self.sample_name or path.name,
            technique=self.technique,
            instrument=self.instrument,
            start_time=first_info.start_time,
            others={
                "file_path": str(path),
                "n_files": len(all_infos),
            },
        )

        return RawData(data=tree), root_info
