import importlib.util
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echemistpy.utils.external.echem.biologic_reader import t_str
from echemistpy.utils.plotting.echemfig import (
    load_example_gpcl_dataset,
    plot_voltage_time,
)

matplotlib.use("Agg")


def _build_dataset() -> xr.Dataset:
    coords = {"time_index": np.arange(5)}
    data_vars = {
        t_str: xr.DataArray(np.linspace(0.0, 2.0, 5), dims=("time_index",), attrs={"unit": "s"}),
        "Ewe/V": xr.DataArray(
            np.linspace(1.0, 1.5, 5), dims=("time_index",), attrs={"unit": "V"}
        ),
    }
    return xr.Dataset(data_vars, coords=coords)


def test_plot_voltage_time_sets_axis_labels():
    dataset = _build_dataset()
    _, axis = plt.subplots()
    axis = plot_voltage_time(dataset, ax=axis, label="cycle 1")
    assert axis.get_xlabel() == f"{t_str} (s)"
    assert axis.get_ylabel() == "Ewe/V (V)"
    assert axis.lines


def test_plot_voltage_time_requires_voltage_trace():
    dataset = _build_dataset().drop_vars("Ewe/V")
    with pytest.raises(ValueError):
        plot_voltage_time(dataset)


GALVANI_AVAILABLE = importlib.util.find_spec("galvani") is not None


@pytest.mark.skipif(not GALVANI_AVAILABLE, reason="galvani is required to parse .mpr files")
def test_load_example_gpcl_dataset_contains_voltage_trace():
    dataset = load_example_gpcl_dataset()
    assert t_str in dataset.variables
    assert "Ewe/V" in dataset.variables
    assert dataset.sizes["time_index"] > 0
