from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echemistpy.external.echem import BiologicMPTReader  # noqa: E402
from echemistpy.external.echem.biologic_reader import t_str  # noqa: E402

ECHEM_DIR = ROOT / "examples" / "echem"


@pytest.mark.parametrize(
    ("filename", "expected_columns"),
    [
        ("Biologic_EIS.mpr", {"freq/Hz", "|Z|/Ohm", "unknown_215"}),
        ("Biologic_GPCL.mpr", {"Ewe/V", "Q charge/discharge/mA.h"}),
    ],
)
def test_biologic_reader_handles_mpr_files(filename, expected_columns):
    reader = BiologicMPTReader()
    measurement = reader.read(ECHEM_DIR / filename)

    data = measurement.data
    assert t_str in data.variables
    for column in expected_columns:
        assert column in data.variables

    extras = measurement.metadata.extras
    assert extras.get("mpr_version") is not None
    assert isinstance(extras.get("mpr_columns"), tuple)
    assert measurement.axes
    assert measurement.axes[0].values is not None
    assert len(measurement.axes[0].values) == data.sizes["time_index"]
