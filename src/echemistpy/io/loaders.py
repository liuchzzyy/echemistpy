"""Generic file loading helpers to keep notebooks lean."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import xarray as xr

Loader = Callable[..., xr.Dataset]


def _structured_array_to_dataset(array: np.ndarray) -> xr.Dataset:
    if array.size == 0:
        return xr.Dataset()
    if array.ndim == 0:
        array = array.reshape(1)
    if array.dtype.names is None:
        raise ValueError("Tabular text files must include a header row.")
    row_dim = np.arange(array.shape[0])
    data_vars = {name: ("row", array[name]) for name in array.dtype.names}
    return xr.Dataset(data_vars=data_vars, coords={"row": row_dim})


def _load_delimited(path: Path, *, delimiter: str, **kwargs: Any) -> xr.Dataset:
    array = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=None, encoding=None, **kwargs)
    return _structured_array_to_dataset(array)


def _load_json_table(path: Path, **_: Any) -> xr.Dataset:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        records: Iterable[Mapping[str, Any]] = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("JSON tables must be a dict or list of dicts.")
    records = list(records)
    if not records:
        return xr.Dataset()
    keys = sorted({key for record in records for key in record})
    data_vars = {
        key: ("row", np.asarray([record.get(key) for record in records], dtype=object))
        for key in keys
    }
    return xr.Dataset(data_vars=data_vars, coords={"row": np.arange(len(records))})


_LOADER_MAP: Dict[str, Loader] = {
    "csv": lambda path, **kwargs: _load_delimited(path, delimiter=",", **kwargs),
    "tsv": lambda path, **kwargs: _load_delimited(path, delimiter="\t", **kwargs),
    "json": _load_json_table,
    "nc": lambda path, **kwargs: xr.open_dataset(path, **kwargs),
    "nc4": lambda path, **kwargs: xr.open_dataset(path, **kwargs),
    "netcdf": lambda path, **kwargs: xr.open_dataset(path, **kwargs),
}


def load_table(path: str | Path, *, fmt: Optional[str] = None, storage_options: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> xr.Dataset:
    """Load a tabular file into an :class:`xarray.Dataset`.

    Examples
    --------
    The helper understands multiple formats; JSON tables make it easy to craft
    doctest-sized fixtures.

    >>> import json, tempfile
    >>> from pathlib import Path
    >>> payload = [
    ...     {"voltage": 3.1, "current": 0.5},
    ...     {"voltage": 3.4, "current": 0.4},
    ... ]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     table_path = Path(tmp) / "trace.json"
    ...     _ = table_path.write_text(json.dumps(payload))
    ...     dataset = load_table(table_path)
    >>> dataset["voltage"].values.tolist()
    [3.1, 3.4]
    >>> dataset.row.size
    2
    """

    del storage_options  # not used yet but kept for API stability

    path = Path(path)
    extension = (fmt or path.suffix.lstrip(".")).lower()
    try:
        loader = _LOADER_MAP[extension]
    except KeyError as exc:
        raise ValueError(f"Unsupported file extension '{extension}'.") from exc
    return loader(path, **kwargs)


def register_loader(extension: str, loader: Loader) -> None:
    """Allow users to plug-in custom loaders (e.g. for proprietary formats)."""

    _LOADER_MAP[extension.lower()] = loader
