"""Helpers for electrochemistry figures and example validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from echemistpy.utils.external.echem import BiologicMPTReader
from echemistpy.utils.external.echem.biologic_reader import t_str

DEFAULT_VOLTAGE_KEY = "Ewe/V"
EXAMPLE_FILENAME = "Biologic_GPCL.mpr"


def load_biologic_dataset(path: str | Path) -> xr.Dataset:
    """Load a BioLogic ``.mpr`` file into an :class:`xarray.Dataset`."""

    reader = BiologicMPTReader()
    measurement = reader.read(Path(path))
    dataset = measurement.data
    _ensure_voltage_time_columns(dataset)
    return dataset


def load_example_gpcl_dataset() -> xr.Dataset:
    """Return the example GPCL dataset shipped with the repository."""

    example_path = _find_example_file(EXAMPLE_FILENAME)
    return load_biologic_dataset(example_path)


def plot_voltage_time(
    dataset: xr.Dataset,
    *,
    time_key: str = t_str,
    voltage_key: str = DEFAULT_VOLTAGE_KEY,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Plot a voltage--time trace from an electrochemistry dataset."""

    axis = ax or plt.gca()
    time_data, voltage_data = _extract_voltage_time_arrays(
        dataset,
        time_key=time_key,
        voltage_key=voltage_key,
    )
    axis.plot(time_data.values, voltage_data.values, label=label)
    axis.set_xlabel(_format_axis_label(time_key, time_data.attrs.get("unit")))
    axis.set_ylabel(_format_axis_label(voltage_key, voltage_data.attrs.get("unit")))
    axis.set_title("Voltage vs Time")
    if label:
        axis.legend()
    return axis


def plot_voltage_time_from_file(
    path: str | Path,
    *,
    time_key: str = t_str,
    voltage_key: str = DEFAULT_VOLTAGE_KEY,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Load a BioLogic file and plot the voltage--time trace."""

    dataset = load_biologic_dataset(path)
    return plot_voltage_time(
        dataset,
        time_key=time_key,
        voltage_key=voltage_key,
        ax=ax,
        label=label,
    )


def _extract_voltage_time_arrays(
    dataset: xr.Dataset,
    *,
    time_key: str,
    voltage_key: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    _ensure_voltage_time_columns(dataset, keys=(time_key, voltage_key))
    return dataset[time_key], dataset[voltage_key]


def _ensure_voltage_time_columns(
    dataset: xr.Dataset,
    *,
    keys: tuple[str, str] | None = None,
) -> None:
    required = keys or (t_str, DEFAULT_VOLTAGE_KEY)
    missing = [key for key in required if key not in dataset.variables]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )


def _format_axis_label(name: str, unit: str | None) -> str:
    if unit:
        return f"{name} ({unit})"
    return name


def _find_example_file(filename: str) -> Path:
    """Locate a file inside ``examples/echem`` relative to the project root."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "examples" / "echem" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to locate {filename!r} relative to {current}."
    )


__all__ = [
    "DEFAULT_VOLTAGE_KEY",
    "EXAMPLE_FILENAME",
    "load_biologic_dataset",
    "load_example_gpcl_dataset",
    "plot_voltage_time",
    "plot_voltage_time_from_file",
]
