"""Comprehensive tests for electrochemistry data readers using real data files.

This test module validates both BioLogic and LANHE readers with actual data from
the examples/echem directory, ensuring the readers correctly parse metadata,
extract time-series data, and handle edge cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echemistpy.io.loaders import load_data_file
from echemistpy.utils.external.echem import BiologicMPTReader
from echemistpy.utils.external.echem.biologic_reader import t_str
from echemistpy.utils.external.echem.lanhe_reader import LanheReader

ECHEM_DIR = ROOT / "examples" / "echem"


class TestBiologicReaderGPCL:
    """Test BioLogic reader with galvanostatic cycling data."""

    @pytest.fixture
    def measurement(self):
        """Load the BioLogic GPCL measurement."""
        reader = BiologicMPTReader()
        return reader.read(ECHEM_DIR / "Biologic_GPCL.mpr")

    def test_basic_structure(self, measurement):
        """Verify measurement has expected structure."""
        assert measurement.data is not None
        assert measurement.metadata is not None
        assert measurement.axes is not None
        assert len(measurement.axes) > 0

    def test_time_column_exists(self, measurement):
        """Verify time column is present."""
        assert t_str in measurement.data.variables
        time_values = measurement.data[t_str].values
        assert len(time_values) > 0
        assert time_values[0] >= 0

    def test_essential_columns(self, measurement):
        """Verify essential electrochemistry columns are present."""
        expected_columns = {
            "Ewe/V",  # Working electrode potential
            "Q charge/discharge/mA.h",  # Charge
        }
        data_vars = set(measurement.data.data_vars.keys())
        for col in expected_columns:
            assert col in data_vars, f"Missing expected column: {col}"
        
        # Verify at least one current-related column exists
        current_cols = [c for c in data_vars if "I" in c or "current" in c.lower()]
        assert len(current_cols) > 0, "Should have at least one current-related column"

    def test_time_monotonic(self, measurement):
        """Verify time values are monotonically increasing."""
        time_values = measurement.data[t_str].values
        differences = np.diff(time_values)
        assert np.all(differences >= 0), "Time values should be monotonically increasing"

    def test_data_consistency(self, measurement):
        """Verify all data arrays have consistent length."""
        time_index_size = measurement.data.sizes["time_index"]
        for var_name, var_data in measurement.data.data_vars.items():
            assert var_data.sizes["time_index"] == time_index_size, (
                f"Variable {var_name} has inconsistent size"
            )

    def test_metadata_fields(self, measurement):
        """Verify metadata contains expected fields."""
        metadata = measurement.metadata
        assert metadata.technique == "EC"
        assert metadata.instrument == "BioLogic EC-Lab"
        assert metadata.sample_name == "Biologic_GPCL.mpr"
        
        # Check extras
        extras = metadata.extras
        assert "mpr_version" in extras
        assert "mpr_columns" in extras
        assert isinstance(extras["mpr_columns"], tuple)

    def test_data_ranges(self, measurement):
        """Verify data values are in reasonable ranges."""
        # Voltage should be reasonable (e.g., 0-5V for typical electrochemistry)
        voltage = measurement.data["Ewe/V"].values
        assert np.all(np.isfinite(voltage))
        assert -10 < voltage.min() < voltage.max() < 10

        # Charge should be finite
        charge = measurement.data["Q charge/discharge/mA.h"].values
        assert np.all(np.isfinite(charge))

    def test_axis_alignment(self, measurement):
        """Verify axis values match time data."""
        axis = measurement.axes[0]
        assert axis.name == t_str
        assert axis.unit == "s"
        time_values = measurement.data[t_str].values
        assert len(axis.values) == len(time_values)
        assert np.allclose(axis.values, time_values)


class TestBiologicReaderEIS:
    """Test BioLogic reader with electrochemical impedance spectroscopy data."""

    @pytest.fixture
    def measurement(self):
        """Load the BioLogic EIS measurement."""
        reader = BiologicMPTReader()
        return reader.read(ECHEM_DIR / "Biologic_EIS.mpr")

    def test_eis_specific_columns(self, measurement):
        """Verify EIS-specific columns are present."""
        expected_columns = {
            "freq/Hz",  # Frequency
            "|Z|/Ohm",  # Impedance magnitude
            "Re(Z)/Ohm",  # Real part of impedance
            "-Im(Z)/Ohm",  # Imaginary part of impedance
            "Phase(Z)/deg",  # Phase
        }
        data_vars = set(measurement.data.data_vars.keys())
        for col in expected_columns:
            assert col in data_vars, f"Missing EIS column: {col}"

    def test_frequency_range(self, measurement):
        """Verify frequency values are positive and reasonable."""
        freq = measurement.data["freq/Hz"].values
        assert np.all(freq > 0), "Frequencies should be positive"
        assert freq.min() > 0.001  # At least 1 mHz
        assert freq.max() < 1e7  # Less than 10 MHz

    def test_impedance_values(self, measurement):
        """Verify impedance values are physically reasonable."""
        z_magnitude = measurement.data["|Z|/Ohm"].values
        assert np.all(z_magnitude > 0), "Impedance magnitude should be positive"
        assert np.all(np.isfinite(z_magnitude))

        # Check that real and imaginary parts are consistent with magnitude
        re_z = measurement.data["Re(Z)/Ohm"].values
        im_z = measurement.data["-Im(Z)/Ohm"].values
        calculated_mag = np.sqrt(re_z**2 + im_z**2)
        assert np.allclose(calculated_mag, z_magnitude, rtol=0.01)

    def test_phase_range(self, measurement):
        """Verify phase values are in valid range."""
        phase = measurement.data["Phase(Z)/deg"].values
        assert np.all(phase >= -90), "Phase should be >= -90 degrees"
        assert np.all(phase <= 90), "Phase should be <= 90 degrees"

    def test_unknown_columns_handled(self, measurement):
        """Verify unknown column IDs are handled gracefully."""
        # The EIS file contains unknown column IDs (215, 175, 176, 177)
        # These should be present as unknown_XXX columns
        data_vars = set(measurement.data.data_vars.keys())
        unknown_cols = [col for col in data_vars if col.startswith("unknown_")]
        assert len(unknown_cols) > 0, "Should have unknown columns from column IDs"


class TestLanheReaderGPCL:
    """Test LANHE reader with galvanostatic cycling data."""

    @pytest.fixture
    def reader(self):
        """Create LANHE reader for GPCL data."""
        return LanheReader(ECHEM_DIR / "LANHE_GPCL.ccs")

    def test_metadata_extraction(self, reader):
        """Verify metadata is correctly extracted."""
        metadata = reader.metadata
        assert isinstance(metadata, dict)
        
        # Check key metadata fields
        assert "test_name" in metadata
        assert metadata["test_name"]  # Should not be empty
        assert "start_date" in metadata
        assert "equipment_id" in metadata

    def test_block_counting(self, reader):
        """Verify block types are correctly counted."""
        counts = reader.block_counts
        assert len(counts) > 0
        
        # Should have data points (tag 0x0603)
        data_blocks = [(tag, ch) for tag, ch in counts if tag == 0x0603]
        assert len(data_blocks) > 0, "Should have data point blocks"

    def test_sample_decoding(self, reader):
        """Verify samples are correctly decoded."""
        samples = reader.samples
        assert len(samples) > 0
        
        # Verify sample structure
        first_sample = samples[0]
        assert hasattr(first_sample, "elapsed_s")
        assert hasattr(first_sample, "values")
        assert len(first_sample.values) == 4

    def test_time_progression(self, reader):
        """Verify time values progress monotonically."""
        times = []
        for sample in reader.iter_samples(tag_filter=0x0603):
            times.append(sample.elapsed_s)
            if len(times) >= 100:  # Sample first 100
                break
        
        assert len(times) > 0
        differences = np.diff(times)
        assert np.all(differences >= 0), "Time should progress monotonically"

    def test_value_ranges(self, reader):
        """Verify decoded values are in reasonable ranges."""
        values_list = []
        for sample in reader.iter_samples(tag_filter=0x0603):
            values_list.append(sample.values)
            if len(values_list) >= 1000:  # Sample first 1000
                break
        
        values_array = np.array(values_list)
        assert values_array.shape[1] == 4
        
        # First value is typically voltage
        voltages = values_array[:, 0]
        assert np.all(np.isfinite(voltages))
        assert -5 < voltages.min() < voltages.max() < 5

    def test_tag_filtering(self, reader):
        """Verify tag filtering works correctly."""
        # Get samples with specific tag
        data_samples = list(reader.iter_samples(tag_filter=0x0603))
        assert len(data_samples) > 0
        assert all(s.tag == 0x0603 for s in data_samples)

    def test_channel_filtering(self, reader):
        """Verify channel filtering works correctly."""
        # Get all channels first
        all_channels = {s.channel_id for s in reader.samples}
        assert len(all_channels) > 0
        
        # Filter by one channel
        channel = list(all_channels)[0]
        filtered = list(reader.iter_samples(channel_filter=channel))
        assert all(s.channel_id == channel for s in filtered)

    def test_csv_export(self, reader, tmp_path):
        """Verify CSV export functionality."""
        csv_file = tmp_path / "test_export.csv"
        reader.export_csv(csv_file, tag_filter=0x0603)

        assert csv_file.exists()
        assert csv_file.stat().st_size > 0
        
        # Verify CSV can be read
        with csv_file.open() as f:
            header = f.readline()
            assert "elapsed_s" in header
            assert "value1" in header

    def test_to_dataset_roundtrip(self, reader):
        """Verify conversion to xarray Dataset preserves key information."""
        dataset = reader.to_dataset()
        assert dataset.sizes["sample_index"] == len(reader.samples)
        assert "lanhe_metadata" in dataset.attrs
        assert "value1" in dataset.data_vars


class TestReaderCompatibility:
    """Test compatibility and edge cases for both readers."""

    def test_biologic_reader_reuse(self):
        """Verify reader can be reused for multiple files."""
        reader = BiologicMPTReader()
        
        # Read first file
        m1 = reader.read(ECHEM_DIR / "Biologic_GPCL.mpr")
        assert m1.data.sizes["time_index"] > 0
        
        # Read second file
        m2 = reader.read(ECHEM_DIR / "Biologic_EIS.mpr")
        assert m2.data.sizes["time_index"] > 0
        
        # Verify they're different
        assert m1.data.sizes["time_index"] != m2.data.sizes["time_index"]

    def test_biologic_metadata_override(self):
        """Verify metadata can be overridden during read."""
        reader = BiologicMPTReader()
        custom_name = "Test Sample"
        custom_extras = {"experiment_id": "EXP-001"}
        
        measurement = reader.read(
            ECHEM_DIR / "Biologic_GPCL.mpr",
            sample_name=custom_name,
            metadata_extras=custom_extras,
        )
        
        assert measurement.metadata.sample_name == custom_name
        assert "experiment_id" in measurement.metadata.extras
        assert measurement.metadata.extras["experiment_id"] == "EXP-001"

    def test_lanhe_reader_multiple_instances(self):
        """Verify multiple reader instances work independently."""
        reader1 = LanheReader(ECHEM_DIR / "LANHE_GPCL.ccs")
        reader2 = LanheReader(ECHEM_DIR / "LANHE_GPCL.ccs")
        
        assert len(reader1.samples) == len(reader2.samples)
        assert reader1.metadata == reader2.metadata


class TestLanheLoaderIntegration:
    """Ensure the generic loader can ingest LANHE binary files."""

    def test_load_data_file_returns_dataset(self):
        dataset = load_data_file(ECHEM_DIR / "LANHE_GPCL.ccs")
        assert dataset.sizes["sample_index"] > 0
        assert "value1" in dataset
        assert dataset.attrs["lanhe_metadata"]["test_name"]

    def test_load_data_file_all_tags(self):
        data_only = load_data_file(ECHEM_DIR / "LANHE_GPCL.ccs", tag_filter=0x0603)
        everything = load_data_file(ECHEM_DIR / "LANHE_GPCL.ccs", tag_filter=None)
        assert everything.sizes["sample_index"] >= data_only.sizes["sample_index"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
