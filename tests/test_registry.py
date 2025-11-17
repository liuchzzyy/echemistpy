from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from echemistpy import analysis
from echemistpy.io import reorganization as io_reorganization
from echemistpy.io import save as io_save
from echemistpy.io.reorganization import DataObject, MeasurementMetadata, MeasurementRecord
from echemistpy.io.reading.base import BaseFileReader
from echemistpy.io.reading.registry import (
    ReaderNotRegisteredError,
    dataset_from_records,
    load_measurement,
    register_reader,
)


class _DummyReader(BaseFileReader):
    technique = "dummy"

    def read(self, path: str | Path, **kwargs):  # noqa: ARG002 - exercise registry plumbing
        data = xr.Dataset(
            data_vars={
                "x": ("index", np.array([1.0, 2.0, 3.0], dtype=float)),
                "current": ("index", np.array([1.0, -2.0, 3.0], dtype=float)),
                "potential": ("index", np.array([0.0, 0.5, 1.0], dtype=float)),
            },
            coords={"index": np.arange(3, dtype=int)},
        )
        metadata = MeasurementMetadata(sample_id=Path(path).stem, technique=self.technique)
        return MeasurementRecord(metadata=metadata, data=data)


register_reader(_DummyReader)


def test_load_measurement(tmp_path):
    file_path = tmp_path / "fake.csv"
    file_path.write_text("x\n1")
    record = load_measurement(file_path, "dummy", sample_id="S1", instrument="Potentiostat")
    assert record.metadata.sample_id == "S1"
    assert record.metadata.instrument == "Potentiostat"
    assert set(record.data.data_vars) == {"x", "current", "potential"}


def test_dataset_from_records(tmp_path):
    file_path = tmp_path / "fake.csv"
    file_path.write_text("x\n1")
    record = load_measurement(file_path, "dummy")
    summary = dataset_from_records([record])
    assert "row_count" in summary.data_vars


def test_missing_reader(tmp_path):
    file_path = tmp_path / "fake.csv"
    file_path.write_text("x\n1")
    try:
        load_measurement(file_path, "unknown")
    except ReaderNotRegisteredError:
        assert True
    else:  # pragma: no cover - defensive programming
        raise AssertionError("Expected ReaderNotRegisteredError")


def test_cleaning_and_analysis_pipeline(tmp_path):
    file_path = tmp_path / "dummy.csv"
    file_path.write_text("stub")
    record = load_measurement(file_path, "dummy")
    cleaned = io_reorganization.rolling_average(record, window=2)
    assert int(cleaned.data.sizes.get("index", 0)) == int(record.data.sizes.get("index", 0))
    peak = analysis.echem.peak_current(cleaned)
    assert peak > 0


def test_xps_analysis():
    metadata = MeasurementMetadata(sample_id="S1", technique="xps")
    dataset = xr.Dataset(
        data_vars={
            "binding_energy": ("index", np.linspace(0, 5, 6)),
            "intensity": ("index", np.array([1, 2, 8, 4, 2, 1], dtype=float)),
        },
        coords={"index": np.arange(6, dtype=int)},
    )
    record = MeasurementRecord(metadata=metadata, data=dataset)
    result = analysis.xps.peak_characteristics(record)
    assert result["peak_intensity"] == 8


def test_record_to_data_object_round_trip(tmp_path):
    metadata = MeasurementMetadata(sample_id="S42", technique="cv")
    dataset = xr.Dataset(
        data_vars={
            "time": ("index", np.array([0.0, 1.0, 2.0], dtype=float)),
            "current": ("index", np.array([0.1, -0.2, 0.3], dtype=float)),
        },
        coords={"index": np.arange(3, dtype=int)},
    )
    record = MeasurementRecord(metadata=metadata, data=dataset, annotations=("baseline",))

    data_object = record.to_data_object()
    assert isinstance(data_object, DataObject)
    assert "metadata" in data_object.to_dataset().attrs

    restored = data_object.to_measurement_record()
    xr.testing.assert_equal(restored.data, record.data)

    output = io_save.save_data_object(data_object, tmp_path / "record.nc")
    loaded = xr.load_dataset(output)
    try:
        metadata_json = loaded.attrs["metadata"]
        metadata_dict = metadata_json if isinstance(metadata_json, dict) else json.loads(metadata_json)
        assert metadata_dict["sample_id"] == "S42"
    finally:
        loaded.close()
