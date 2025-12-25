# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""XAS Data Reader for ALBA CLAESS beamline files."""

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from traitlets import HasTraits, List, Unicode

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class CLAESSReader(HasTraits):
    """Reader for CLAESS XAS .dat files.

    Supports reading a single .dat file (which may contain multiple scans)
    or a directory containing multiple files.
    Filters files to only include those without digits in their names.
    """

    filepath = Unicode()
    sample_name = Unicode(None, allow_none=True)
    technique = List(Unicode(), default_value=["xas", "operando"])
    instrument = Unicode("ALBA_CLAESS", allow_none=True)
    selected_columns = List(Unicode(), default_value=["energyc", "a_i0_1", "a_i0_2", "a_i1_1", "a_i1_2", "absorption"], help="Columns to keep when cleaning.")

    def __init__(self, filepath: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def parse_date(self, date_str: str) -> datetime:
        """Parse SPEC date format: Thu Dec 11 12:52:40 2025."""
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y")

    def load(self, edges: list[str] | None = None, **kwargs) -> tuple[RawData, RawDataInfo]:
        """Load CLAESS file(s) and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath not set")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return self._load_single_file(path)
        elif path.is_dir():
            return self._load_directory(path, edges=edges)
        else:
            raise ValueError(f"Path is neither a file nor a directory: {path}")

    def _clean_data(self, data: xr.Dataset | xr.DataTree) -> xr.Dataset | xr.DataTree:
        """Keep only specific columns defined in selected_columns."""
        if isinstance(data, xr.Dataset):
            existing_cols = [c for c in self.selected_columns if c in data.data_vars or c in data.coords]
            return data[existing_cols]
        elif isinstance(data, xr.DataTree):
            # For DataTree, we need to create a new tree with filtered datasets
            new_dict = {}
            for path, node in data.subtree:
                if node.dataset is not None:
                    existing_cols = [c for c in self.selected_columns if c in node.dataset.data_vars or c in node.dataset.coords]
                    new_dict[path] = node.dataset[existing_cols]
            return xr.DataTree.from_dict(new_dict, name=data.name)
        return data

    def _load_single_file(self, path: Path) -> tuple[RawData, RawDataInfo]:
        """Internal method to load a single CLAESS file."""
        if path.suffix.lower() == ".dat":
            data, metadata = self._read_spec_file(path)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        # Automatically clean data
        data = self._clean_data(data)

        if isinstance(data, xr.Dataset):
            n_records = len(data.record) if "record" in data.dims else 1
        else:
            # DataTree
            n_records = len(data.children)

        raw_info = RawDataInfo(
            sample_name=self.sample_name or path.stem,
            technique=self.technique,
            instrument=self.instrument,
            start_time=metadata.get("start_time"),
            others={
                "file_path": str(path),
                "n_files": n_records,
            },
        )

        # Set standardized attributes
        data.attrs = {
            "file_name": [path.stem],
            "n_files": n_records,
            "structure": "Dataset" if isinstance(data, xr.Dataset) else "DataTree",
        }

        return RawData(data=data), raw_info

    def _read_spec_file(self, path: Path) -> tuple[xr.Dataset | xr.DataTree, dict]:
        """Parse a SPEC-like .dat file with multiple scans."""
        datasets, scan_times, header = self._parse_spec_file(path)
        merged = self._merge_scans(datasets, scan_times, path.stem)

        start_time = None
        if isinstance(merged, xr.Dataset):
            start_time = merged.attrs.get("start_time")

        return merged, {"header": header, "start_time": start_time}

    def _parse_spec_file(self, path: Path) -> tuple[dict[str, xr.Dataset], dict[str, datetime], str]:
        """Internal method to parse SPEC file into raw datasets and times."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split by #S lines
        scans_raw = re.split(r"^#S\s+", content, flags=re.MULTILINE)
        header = scans_raw[0]
        scans_raw = scans_raw[1:]

        if not scans_raw:
            # Try to read as a simple table if no #S found
            try:
                df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
                ds = df.to_xarray().rename({"index": "point"})
                return {"scan_1": ds}, {}, header
            except Exception as e:
                raise ValueError(f"No scans found and failed to read as table in {path}: {e}")

        datasets = {}
        scan_times = {}
        for scan_content in scans_raw:
            lines = scan_content.splitlines()
            if not lines:
                continue

            scan_line = lines[0]
            parts = scan_line.split()
            scan_id = parts[0] if parts else "unknown"

            data_lines = []
            columns = []

            for line in lines[1:]:
                if line.startswith("#L"):
                    columns = line[3:].strip().split()
                elif line.startswith("#D"):
                    date_str = line[3:].strip()
                    try:
                        scan_times[f"scan_{scan_id}"] = self.parse_date(date_str)
                    except Exception:
                        pass
                elif line.startswith("#"):
                    continue
                elif line.strip():
                    data_lines.append(line.split())

            if not data_lines:
                continue

            if not columns:
                columns = [f"col_{i}" for i in range(len(data_lines[0]))]

            try:
                df = pd.DataFrame(data_lines, columns=columns).astype(float)
                # Calculate absorption: ln((a_i0_1+a_i0_2)/(a_i1_1+a_i1_2))
                if all(c in df.columns for c in ["a_i0_1", "a_i0_2", "a_i1_1", "a_i1_2"]):
                    i0 = df["a_i0_1"] + df["a_i0_2"]
                    i1 = df["a_i1_1"] + df["a_i1_2"]
                    df["absorption"] = np.log(i0 / i1)

                # Use energyc as the primary index/dimension
                if "energyc" in df.columns:
                    # Drop duplicates to ensure valid index
                    df = df.drop_duplicates(subset=["energyc"]).set_index("energyc")

                ds = df.to_xarray()
                datasets[f"scan_{scan_id}"] = ds
            except Exception as e:
                logger.warning("Failed to parse scan %s in %s: %s", scan_id, path, e)

        return datasets, scan_times, header

    def _merge_scans(self, datasets: dict[str, xr.Dataset], scan_times: dict[str, datetime], name: str) -> xr.Dataset | xr.DataTree:
        """Internal method to merge multiple scan datasets into one."""
        if not datasets:
            return None

        # Extract start_time from the first scan that has a date
        start_time = None
        first_scan_id = list(datasets.keys())[0]
        if first_scan_id in scan_times:
            start_time = scan_times[first_scan_id].strftime("%Y-%m-%d %H:%M:%S")

        if len(datasets) > 1:
            try:
                ds_list = list(datasets.values())

                # Find a common energy grid for interpolation
                # We use the scan with the most points as the reference grid
                all_energies = [ds.energyc.values for ds in ds_list if "energyc" in ds.coords]
                if all_energies:
                    ref_energy = max(all_energies, key=len)

                    interpolated_list = []
                    for ds in ds_list:
                        if "energyc" in ds.coords:
                            # Interpolate all data variables onto the reference energy grid
                            new_vars = {}
                            for var in ds.data_vars:
                                # Use interp1d to interpolate data
                                f = interp1d(ds.energyc.values, ds[var].values, bounds_error=False, fill_value=np.nan)
                                new_vars[var] = (("energyc"), f(ref_energy))

                            new_ds = xr.Dataset(new_vars, coords={"energyc": ref_energy})
                            interpolated_list.append(new_ds)
                        else:
                            interpolated_list.append(ds)
                    ds_list = interpolated_list

                # Try to concatenate along 'record' dimension
                combined = xr.concat(ds_list, dim="record")
                # Set record values from 1
                scan_ids = list(datasets.keys())
                combined = combined.assign_coords(record=np.arange(1, len(datasets) + 1))

                # Add file_name (formerly scan) coordinate
                combined = combined.assign_coords(file_name=("record", [name] * len(datasets)))

                # Add systime and time_s coordinates
                if scan_times:
                    systimes = pd.to_datetime([scan_times.get(sid) for sid in scan_ids])
                    combined.coords["systime"] = ("record", systimes)

                    # Calculate relative time as timedelta
                    valid_times = systimes[systimes.notnull()]
                    if not valid_times.empty:
                        t0 = valid_times[0]
                        time_deltas = systimes - t0
                        combined.coords["time_s"] = ("record", time_deltas)

                combined.attrs["start_time"] = start_time
                return combined
            except Exception as e:
                logger.warning("Failed to merge scans for %s: %s. Returning DataTree instead.", name, e)

        if len(datasets) == 1:
            ds = list(datasets.values())[0]
            # Expand dims to make record a dimension
            ds = ds.expand_dims("record")
            ds = ds.assign_coords(record=[1])
            ds = ds.assign_coords(file_name=("record", [name]))

            sid = list(datasets.keys())[0]
            if sid in scan_times:
                ds.coords["systime"] = ("record", pd.to_datetime([scan_times[sid]]))
                ds.coords["time_s"] = ("record", pd.to_timedelta([0], unit="s"))

            ds.attrs["start_time"] = start_time
            return ds
        else:
            dt = xr.DataTree.from_dict(datasets, name=name)
            return dt

    def _auto_detect_edges_from_folders(self, files: list[Path], root_path: Path) -> list[str] | None:
        """从文件的上一级文件夹名称中自动检测 edges。

        Args:
            files: 相关的 .dat 文件列表
            root_path: 根目录路径

        Returns:
            检测到的 edge 列表（按长度降序排列）或 None
        """
        # 元素周期表
        elements = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
        ]

        detected = set()
        for f in files:
            # 获取文件的上一级文件夹名（相对于根路径）
            folder_name = f.parent.name

            # 在文件夹名中查找元素符号
            for el in elements:
                # 匹配模式：
                # 1. 在开头或前面是非字母数字字符
                # 2. 后面是大写字母（如 MnFoil）或非字母数字字符或结尾
                # 这样可以避免误匹配，如 'P1' 中的 'P' 或 'C1' 中的 'C'
                pattern = rf"(^|[^a-zA-Z0-9]){el}([A-Z]|[^a-zA-Z0-9]|$)"
                if re.search(pattern, folder_name):
                    detected.add(el)

        if detected:
            # 按长度降序排列，以便优先匹配较长的符号（如 Mn 优先于 N）
            return sorted(list(detected), key=lambda x: (-len(x), x))

        return None

    def _load_directory(self, path: Path, edges: list[str] | None = None) -> tuple[RawData, RawDataInfo]:
        """Load all relevant files in a directory."""
        # Find all .dat files
        all_files = list(path.rglob("*.dat"))

        # Filter: skip files that look like individual scan exports (e.g., *_000.dat)
        # These usually have exactly 3 digits at the end of the filename
        relevant_files = [f for f in all_files if not re.search(r"_\d{3}$", f.stem)]

        if not relevant_files:
            raise FileNotFoundError(f"No relevant .dat files found in {path}")

        # Auto-detect edges if not provided (从文件的上一级文件夹名称中提取)
        if edges is None:
            edges = self._auto_detect_edges_from_folders(relevant_files, path)
            if edges:
                logger.info("Auto-detected edges from folder names: %s", edges)

        tree = xr.DataTree(name=path.name)
        all_infos = []

        if edges:
            # Group by (edge, clean_rel_path)
            groups = {}
            for f in relevant_files:
                rel_parent = f.parent.relative_to(path)
                matched_edge = None

                # 从文件的上一级文件夹名称中提取 edge
                folder_name = f.parent.name
                for edge in edges:
                    # 使用严格的模式匹配，避免误匹配
                    pattern = rf"(^|[^a-zA-Z0-9]){edge}([A-Z]|[^a-zA-Z0-9]|$)"
                    if re.search(pattern, folder_name):
                        matched_edge = edge
                        break  # 找到第一个匹配的 edge 即可

                if matched_edge:
                    # Strip any folder named 'edge' from the path parts to get clean_rel_path
                    parts = [p for p in rel_parent.parts if p.lower() != matched_edge.lower()]
                    clean_rel_path = "/".join(parts)
                    key = (matched_edge, clean_rel_path)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(f)

            for (edge, clean_rel_path), files in groups.items():
                all_datasets = {}
                all_scan_times = {}
                for f in sorted(files):
                    try:
                        ds_dict, st_dict, _ = self._parse_spec_file(f)
                        for k, v in ds_dict.items():
                            all_datasets[f"{f.stem}_{k}"] = v
                        for k, v in st_dict.items():
                            all_scan_times[f"{f.stem}_{k}"] = v
                    except Exception as e:
                        logger.warning("Failed to parse %s: %s", f, e)

                if not all_datasets:
                    continue

                merged_ds = self._merge_scans(all_datasets, all_scan_times, edge)
                merged_ds = self._clean_data(merged_ds)

                # Determine a name for this dataset node (leaf)
                if len(files) == 1:
                    ds_name = files[0].stem
                else:
                    # Find common prefix of stems for a descriptive name
                    stems = [f.stem for f in files]
                    import os

                    common = os.path.commonprefix(stems).rstrip("_")
                    ds_name = common if len(common) > 2 else "merged_data"

                # Build node path: edge/clean_rel_path/ds_name
                # We ensure the dataset is always a leaf node to avoid AlignmentError
                if clean_rel_path:
                    full_node_name = f"{edge}/{clean_rel_path}/{ds_name}"
                else:
                    full_node_name = f"{edge}/{ds_name}"

                print(f"DEBUG: Adding node {full_node_name} with energyc size {merged_ds.energyc.size}")
                try:
                    tree[full_node_name] = merged_ds
                except Exception as e:
                    # If alignment fails, it's usually because a parent node already has data
                    # or incompatible coordinates. We try to add it with a suffix.
                    logger.warning("Alignment failed for %s: %s. Skipping or renaming.", full_node_name, e)
                    try:
                        tree[f"{full_node_name}_data"] = merged_ds
                    except:
                        pass

                # Create info
                n_records = len(merged_ds.record) if "record" in merged_ds.dims else 1
                info = RawDataInfo(
                    sample_name=ds_name,
                    technique=self.technique,
                    instrument=self.instrument,
                    start_time=merged_ds.attrs.get("start_time"),
                    others={"n_files": n_records},
                )
                all_infos.append(info)
        else:
            for f in sorted(relevant_files):
                try:
                    raw_data, raw_info = self._load_single_file(f)

                    # Determine relative path for the tree node
                    rel_path = f.relative_to(path)
                    # Replace separators and remove extension for node name
                    node_name = str(rel_path.with_suffix("")).replace("\\", "_").replace("/", "_")

                    if isinstance(raw_data.data, xr.DataTree):
                        # If it's already a tree (multi-scan), add its children under a group
                        for name, child in raw_data.data.children.items():
                            tree[f"{node_name}/{name}"] = child
                    else:
                        tree[node_name] = raw_data.data

                    all_infos.append(raw_info)
                except Exception as e:
                    logger.warning("Failed to load file %s: %s", f, e)

        if not tree.children and not tree.has_data:
            raise RuntimeError(f"Failed to load any relevant files from {path}")

        # Merge infos
        root_info = self._merge_infos(all_infos, path)

        # Set standardized attributes
        tree.attrs = {
            "file_name": [info.sample_name for info in all_infos],
            "n_files": root_info.get("n_files"),
            "structure": "DataTree",
        }

        return RawData(data=tree), root_info

    def _merge_infos(self, infos: list[RawDataInfo], root_path: Path) -> RawDataInfo:
        """Merge multiple RawDataInfo objects into one."""
        if not infos:
            return RawDataInfo()

        base = infos[0]
        all_techs = set()
        total_files = 0
        for info in infos:
            for t in info.technique:
                all_techs.add(t)

            n = info.get("n_files")
            if n is not None:
                total_files += n
            else:
                total_files += 1

        return RawDataInfo(
            sample_name=self.sample_name or root_path.name,
            technique=list(all_techs),
            instrument=base.instrument,
            start_time=base.start_time,
            others={"n_files": total_files, "root_path": str(root_path), "structure": "DataTree" if total_files > 1 else "Dataset"},
        )
