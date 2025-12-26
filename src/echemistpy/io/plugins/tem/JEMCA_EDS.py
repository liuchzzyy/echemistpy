# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""TEM EDS Data Reader for JEMCA EMD files."""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from traitlets import HasTraits, List, Unicode

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class JEMCAEDSReader(HasTraits):
    """Reader for JEMCA TEM EDS .emd files."""

    filepath = Unicode()
    sample_name = Unicode(None, allow_none=True)
    technique = List(Unicode(), default_value=["tem", "eds"])
    instrument = Unicode("JEMCA", allow_none=True)

    def __init__(self, filepath: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self, **_kwargs) -> tuple[RawData, RawDataInfo]:
        """Load JEMCA EMD file and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath not set")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        with h5py.File(path, "r") as f:
            tree = xr.DataTree(name=path.stem)

            # 1. Load Images
            if "Data/Image" in f:
                image_group = f["Data/Image"]
                for guid in image_group:
                    try:
                        ds = self._load_dataset(image_group[guid], "image")
                        if ds is not None:
                            name = ds.attrs.get("detector", guid)
                            tree[f"images/{name}"] = ds
                    except Exception as e:
                        logger.warning("Failed to load image %s: %s", guid, e)

            # 2. Load Spectra
            if "Data/Spectrum" in f:
                spec_group = f["Data/Spectrum"]
                for guid in spec_group:
                    try:
                        ds = self._load_dataset(spec_group[guid], "spectrum")
                        if ds is not None:
                            name = ds.attrs.get("detector", guid)
                            tree[f"spectra/{name}"] = ds
                    except Exception as e:
                        logger.warning("Failed to load spectrum %s: %s", guid, e)

            # 3. Load Spectrum Images (Hypercubes)
            if "Data/SpectrumImage" in f:
                si_group = f["Data/SpectrumImage"]
                for guid in si_group:
                    try:
                        if "Data" in si_group[guid]:
                            ds = self._load_dataset(si_group[guid], "spectrum_image")
                            if ds is not None:
                                name = ds.attrs.get("detector", guid)
                                tree[f"spectrum_images/{name}"] = ds
                    except Exception as e:
                        logger.warning("Failed to load spectrum image %s: %s", guid, e)

            # Metadata
            raw_info = RawDataInfo(
                sample_name=self.sample_name or path.stem,
                technique=self.technique,
                instrument=self.instrument,
                others={
                    "file_path": str(path),
                },
            )

            return RawData(data=tree), raw_info

    def _load_dataset(self, group: h5py.Group, data_type: str) -> xr.Dataset | None:
        """Internal method to load a single dataset from an EMD group."""
        if "Data" not in group or "Metadata" not in group:
            return None

        guid = group.name.split("/")[-1]
        print(f"  Loading dataset {guid} (type: {data_type}, shape: {group['Data'].shape})")
        data = group["Data"][:]
        print(f"    Data loaded, shape: {data.shape}")
        metadata = self._parse_metadata(group["Metadata"])

        binary_result = metadata.get("BinaryResult", {})
        detector = binary_result.get("Detector", "Unknown")
        pixel_size = binary_result.get("PixelSize", {})

        dx = float(pixel_size.get("width", 1.0))
        dy = float(pixel_size.get("height", 1.0))

        if data_type == "image":
            if data.ndim == 3:
                if data.shape[2] == 1:
                    data = data[:, :, 0]
                    ny, nx = data.shape
                    dims = ["y", "x"]
                    coords = {"y": np.arange(ny) * dy, "x": np.arange(nx) * dx}
                else:
                    # Stack of images (H, W, N) -> (N, H, W)
                    data = np.moveaxis(data, -1, 0)
                    nf, ny, nx = data.shape
                    dims = ["frame", "y", "x"]
                    coords = {
                        "frame": np.arange(nf),
                        "y": np.arange(ny) * dy,
                        "x": np.arange(nx) * dx,
                    }
            else:
                ny, nx = data.shape
                dims = ["y", "x"]
                coords = {"y": np.arange(ny) * dy, "x": np.arange(nx) * dx}

            ds = xr.Dataset(
                data_vars={"intensity": (dims, data)},
                coords=coords,
                attrs={"detector": detector, "units": "m"},
            )
            print(f"    Image dataset created for {guid}")

        elif data_type == "spectrum":
            if data.ndim == 2 and data.shape[1] == 1:
                data = data[:, 0]

            n_channels = len(data)

            dispersion = 10.0
            offset = 0.0

            # Search in metadata for energy calibration
            acquisition = metadata.get("Acquisition", {})
            if "Dispersion" in acquisition:
                dispersion = float(acquisition["Dispersion"])

            detectors_meta = metadata.get("Detectors", {})
            if detector in detectors_meta:
                det_meta = detectors_meta[detector]
                if "SpectrumBeginEnergy" in det_meta:
                    offset = float(det_meta["SpectrumBeginEnergy"].get("value", 0.0))

            energy = offset + np.arange(n_channels) * dispersion

            ds = xr.Dataset(data_vars={"counts": (["energy"], data)}, coords={"energy": energy}, attrs={"detector": detector, "units": "eV"})
            print(f"    Spectrum dataset created for {guid}")

        elif data_type == "spectrum_image":
            if data.ndim == 3:
                ny, nx, ne = data.shape
                x_coords = np.arange(nx) * dx
                y_coords = np.arange(ny) * dy
                energy = np.arange(ne) * 10.0

                ds = xr.Dataset(data_vars={"counts": (["y", "x", "energy"], data)}, coords={"y": y_coords, "x": x_coords, "energy": energy}, attrs={"detector": detector})
            else:
                return None
        else:
            return None

        ds.attrs.update(metadata)
        return ds

    def _parse_metadata(self, dataset: h5py.Dataset) -> dict:
        """Parse EMD metadata dataset (JSON encoded byte array)."""
        try:
            data = dataset[:]
            if data.ndim == 2:
                data = data[:, 0]
            meta_str = "".join([chr(b) for b in data if b != 0])
            return json.loads(meta_str)
        except Exception:
            return {}
