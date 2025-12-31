# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""TXM Data Reader for MISTRAL beamline HDF5 files."""

import contextlib
import logging
import re
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Tuple, Union

import h5py
import numpy as np
import xarray as xr
from traitlets import HasTraits, Unicode
from traitlets import List as TList

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class MISTRALReader(HasTraits):
    """Reader for MISTRAL TXM .hdf5 files."""

    # --- Constants ---
    INSTRUMENT_NAME: ClassVar[str] = "ALBA_MISTRAL"
    DEFAULT_TECHNIQUE: ClassVar[List[str]] = ["txm", "ex situ"]
    DATE_REGEX: ClassVar[re.Pattern] = re.compile(r"(\d{8})")

    # --- Loader Metadata ---
    supports_directories: ClassVar[bool] = False
    instrument: ClassVar[str] = "mistral"

    # --- Traitlets ---
    filepath = Unicode()
    sample_name = Unicode(None, allow_none=True)
    technique = TList(Unicode(), default_value=DEFAULT_TECHNIQUE)
    # instrument traitlet removed to avoid conflict with ClassVar

    def __init__(self, filepath: Optional[Union[str, Path]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self, **_kwargs: Any) -> Tuple[RawData, RawDataInfo]:
        """Load MISTRAL HDF5 file(s) and return RawData and RawDataInfo."""
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
        """Internal method to load a single MISTRAL HDF5 file."""
        with h5py.File(path, "r") as f:
            if "SpecNormalized" not in f:
                raise ValueError(f"File {path} does not contain 'SpecNormalized' group.")

            group = f["SpecNormalized"]
            ds = self._extract_dataset(group)

            # Extract date from filename
            start_time = self._extract_date(path.name)

            # Metadata
            raw_info = RawDataInfo(
                sample_name=self.sample_name or path.stem,
                technique=list(self.technique),
                instrument=self.instrument,
                start_time=start_time,
                others={"file_path": str(path)},
            )

            return RawData(data=ds), raw_info

    def _extract_dataset(self, group: h5py.Group) -> xr.Dataset:
        """Extract data from HDF5 group and create xarray.Dataset."""
        data_cube = group["spectroscopy_normalized_aligned"][:]
        energy = group["energy"][:]
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

        self._apply_standard_attrs(ds)
        return ds

    @staticmethod
    def _apply_standard_attrs(ds: xr.Dataset) -> None:
        """Apply standard units and long names."""
        ds.energy.attrs.update({"units": "eV", "long_name": "Energy"})
        ds.x.attrs.update({"units": "nm", "long_name": "X Position"})
        ds.y.attrs.update({"units": "nm", "long_name": "Y Position"})
        ds.transmission.attrs.update({"units": "a.u.", "long_name": "Transmission"})
        ds.optical_density.attrs.update({"units": "a.u.", "long_name": "Optical Density"})
        if "rotation_angle" in ds:
            ds.rotation_angle.attrs.update({"units": "deg", "long_name": "Rotation Angle"})

    def _extract_date(self, filename: str) -> Optional[str]:
        """Extract date from filename (e.g., 20230701)."""
        match = self.DATE_REGEX.search(filename)
        if match:
            with contextlib.suppress(Exception):
                d = match.group(1)
                return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        return None

    def _load_directory(self, path: Path) -> Tuple[RawData, RawDataInfo]:
        """Load all MISTRAL HDF5 files in a directory into a DataTree."""
        all_files = sorted(path.rglob("*.hdf5*"))
        if not all_files:
            raise FileNotFoundError(f"No HDF5 files found in {path}")

        tree_dict = {}
        all_infos = []

        for f in all_files:
            with contextlib.suppress(Exception):
                raw_data, raw_info = self._load_single_file(f)
                rel_path = f.relative_to(path)
                # Use the first part of the filename as node name
                file_stem = rel_path.name.split(".")[0]
                node_path = "/" + "/".join([*list(rel_path.parts[:-1]), file_stem])

                tree_dict[node_path] = raw_data.data
                all_infos.append(raw_info)

        if not tree_dict:
            raise RuntimeError(f"Failed to load any files from {path}")

        tree = xr.DataTree.from_dict(tree_dict, name=path.name)
        first_info = all_infos[0] if all_infos else RawDataInfo()
        root_info = RawDataInfo(
            sample_name=self.sample_name or path.name,
            technique=list(self.technique),
            instrument=self.instrument,
            start_time=first_info.start_time,
            others={"n_files": len(all_infos), "structure": "DataTree"},
        )

        return RawData(data=tree), root_info
