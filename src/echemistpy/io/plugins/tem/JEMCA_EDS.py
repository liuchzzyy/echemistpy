# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""TEM EDS Data Reader for JEMCA EMD files."""

import json
import logging
from pathlib import Path
from typing import Any, Tuple

import h5py
import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None
from traitlets import Dict, HasTraits, List, Unicode

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
        self.original_metadata = {}
        self._guid_to_name = {}

    def load(self, **_kwargs) -> tuple[RawData, RawDataInfo]:
        """Load JEMCA EMD file and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath not set")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        with h5py.File(path, "r") as f:
            # Parse global metadata groups
            # Some versions have Displays under Presentation
            metadata_paths = [
                ("Displays", "Displays"),
                ("Presentation/Displays", "Displays"),
                ("Operations", "Operations"),
                ("SharedProperties", "SharedProperties"),
                ("Features", "Features"),
            ]
            for h5_path, meta_key in metadata_paths:
                if h5_path in f:
                    self._parse_global_metadata_group(f[h5_path], meta_key)

            # Build GUID to Name mapping
            self._build_guid_mapping(f)

            tree = xr.DataTree(name=path.stem)

            # 1. Load Images
            if "Data/Image" in f:
                image_group = f["Data/Image"]
                for guid in image_group:
                    try:
                        ds = self._load_dataset(image_group[guid], "image")
                        if ds is not None:
                            name = self._guid_to_name.get(guid, guid)
                            name = name.replace("/", "_").replace(" ", "_")
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
                            name = self._guid_to_name.get(guid, guid)
                            name = name.replace("/", "_").replace(" ", "_")
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
                                name = self._guid_to_name.get(guid, guid)
                                name = name.replace("/", "_").replace(" ", "_")
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
            # Add metadata to others without triggering traitlets comparison issues
            if "SharedProperties" in self.original_metadata:
                raw_info.others["instrument_details"] = self.original_metadata["SharedProperties"]

            return RawData(data=tree), raw_info

    def _parse_global_metadata_group(self, group: h5py.Group, group_name: str):
        """Parse global metadata groups like Displays, Operations, etc."""
        d = {}
        for key in group:
            subgroup = group[key]
            try:
                if isinstance(subgroup, h5py.Group):
                    sub_dict = {}
                    for subkey in subgroup:
                        val = subgroup[subkey]
                        if isinstance(val, h5py.Dataset):
                            sub_dict[subkey] = self._parse_json_dataset(val)
                    d[key] = sub_dict
                elif isinstance(subgroup, h5py.Dataset):
                    d[key] = self._parse_json_dataset(subgroup)
            except Exception as e:
                logger.debug("Failed to parse metadata group %s/%s: %s", group_name, key, e)
        self.original_metadata[group_name] = d

    def _build_guid_mapping(self, f: h5py.File):
        """Build a mapping from GUID to human-readable names."""
        # 1. Check Displays
        if "Displays" in self.original_metadata:
            displays = self.original_metadata["Displays"]
            if "ImageDisplay" in displays:
                for disp_meta in displays["ImageDisplay"].values():
                    if isinstance(disp_meta, dict) and "data" in disp_meta:
                        try:
                            ref_path = disp_meta["data"]
                            if ref_path in f:
                                ref_data = self._parse_json_dataset(f[ref_path])
                                if ref_data and "dataPath" in ref_data:
                                    guid = ref_data["dataPath"].split("/")[-1]
                                    self._guid_to_name[guid] = disp_meta.get("title", guid)
                        except Exception:
                            continue

        # 2. Check Operations (StemInputOperation, EDSInputOperation)
        if "Operations" in self.original_metadata:
            ops = self.original_metadata["Operations"]
            # STEM detectors
            if "StemInputOperation" in ops:
                for op_meta in ops["StemInputOperation"].values():
                    if isinstance(op_meta, dict) and "dataPath" in op_meta:
                        guid = op_meta["dataPath"].split("/")[-1]
                        self._guid_to_name[guid] = op_meta.get("detector", guid)

            # EDS detectors
            if "EDSInputOperation" in ops:
                for op_meta in ops["EDSInputOperation"].values():
                    if isinstance(op_meta, dict) and "dataPath" in op_meta:
                        guid = op_meta["dataPath"].split("/")[-1]
                        self._guid_to_name[guid] = op_meta.get("detector", guid)

    def _parse_json_dataset(self, dataset: h5py.Dataset) -> Any:
        """Parse a dataset containing JSON encoded byte array or object."""
        try:
            data = dataset[:]
            if data.dtype.kind == "O":  # Object type (bytes)
                if isinstance(data[0], bytes):
                    meta_str = data[0].decode("utf-8")
                else:
                    meta_str = str(data[0])
            else:
                if data.ndim == 2:
                    data = data[:, 0]
                # Filter out null bytes and decode
                meta_str = data.tobytes().decode("utf-8").strip("\x00")
            return json.loads(meta_str)
        except Exception:
            return None

    def _load_dataset(self, group: h5py.Group, data_type: str) -> xr.Dataset | None:
        """Internal method to load a single dataset from an EMD group."""
        if "Data" not in group or "Metadata" not in group:
            return None

        guid = group.name.split("/")[-1]
        h5_data = group["Data"]

        # Use dask for large datasets to avoid memory issues and hangs
        if da is not None and h5_data.size > 1e7:  # > 10M elements
            data = da.from_array(h5_data, chunks="auto")
        else:
            data = h5_data[:]

        metadata = self._parse_json_dataset(group["Metadata"])
        if metadata is None:
            metadata = {}

        binary_result = metadata.get("BinaryResult", {})
        detector = binary_result.get("Detector", "Unknown")
        pixel_size = binary_result.get("PixelSize", {})

        # Default to 1.0 if not found
        dx = float(pixel_size.get("width", 1.0))
        dy = float(pixel_size.get("height", 1.0))
        unit = binary_result.get("PixelUnitX", "m")

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
                attrs={"detector": detector, "units": unit},
            )

        elif data_type == "spectrum":
            if data.ndim == 2 and data.shape[1] == 1:
                data = data[:, 0]

            n_channels = len(data)
            dispersion, offset, energy_unit = self._get_energy_calibration(metadata, detector)

            energy = offset + np.arange(n_channels) * dispersion
            ds = xr.Dataset(
                data_vars={"counts": (["energy"], data)},
                coords={"energy": energy},
                attrs={"detector": detector, "units": energy_unit},
            )

        elif data_type == "spectrum_image":
            if data.ndim == 3:
                ny, nx, ne = data.shape
                x_coords = np.arange(nx) * dx
                y_coords = np.arange(ny) * dy

                dispersion, offset, energy_unit = self._get_energy_calibration(metadata, detector)
                energy = offset + np.arange(ne) * dispersion

                ds = xr.Dataset(
                    data_vars={"counts": (["y", "x", "energy"], data)},
                    coords={"y": y_coords, "x": x_coords, "energy": energy},
                    attrs={"detector": detector, "energy_units": energy_unit, "spatial_units": unit},
                )
            else:
                return None
        else:
            return None

        ds.attrs.update(metadata)
        return ds

    def _get_energy_calibration(self, local_metadata: dict, detector: str) -> tuple[float, float, str]:
        """Get energy calibration (dispersion, offset) from metadata."""
        dispersion = 10.0  # Default 10 eV
        offset = 0.0
        unit = "eV"

        # 1. Try local metadata
        acquisition = local_metadata.get("Acquisition", {})
        if "Dispersion" in acquisition:
            dispersion = float(acquisition["Dispersion"])

        # 2. Try global metadata (Detectors group)
        if "SharedProperties" in self.original_metadata:
            detectors = self.original_metadata["SharedProperties"].get("Detectors", {})
            for det_id, det_meta in detectors.items():
                if detector in det_meta.get("DetectorName", ""):
                    dispersion = float(det_meta.get("Dispersion", dispersion))
                    offset = float(det_meta.get("OffsetEnergy", offset))
                    break

        # Velox often uses eV for dispersion but sometimes keV.
        # If dispersion is very small (e.g. 0.01), it might be keV.
        if dispersion < 1.0:
            unit = "keV"

        return dispersion, offset, unit

    def _parse_metadata(self, dataset: h5py.Dataset) -> dict:
        """Parse EMD metadata dataset (JSON encoded byte array)."""
        return self._parse_json_dataset(dataset) or {}
