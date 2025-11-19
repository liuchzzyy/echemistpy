"""Unit tests for structures.py helper methods."""

import pytest
import numpy as np
import xarray as xr

from echemistpy.io.structures import (
    RawData,
    RawDataInfo,
    Measurement,
    MeasurementMetadata,
    AnalysisResult,
    AnalysisResultInfo,
    NXField,
    NXGroup,
    # Test backward compatibility aliases
    RawMetadata,
    MeasurementInfo,
    Results,
    ResultsInfo,
)


class TestRawDataInfo:
    """Tests for RawDataInfo helper methods."""
    
    def test_to_dict(self):
        """Test to_dict method."""
        info = RawDataInfo(meta={"key1": "value1", "key2": 123})
        result = info.to_dict()
        assert result == {"key1": "value1", "key2": 123}
    
    def test_get(self):
        """Test get method."""
        info = RawDataInfo(meta={"technique": "CV", "instrument": "Test"})
        assert info.get("technique") == "CV"
        assert info.get("missing") is None
        assert info.get("missing", "default") == "default"
    
    def test_update(self):
        """Test update method."""
        info = RawDataInfo(meta={"key1": "value1"})
        info.update({"key2": "value2", "key3": 123})
        assert info.meta == {"key1": "value1", "key2": "value2", "key3": 123}


class TestRawData:
    """Tests for RawData helper methods."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample xarray.Dataset for testing."""
        return xr.Dataset({
            "voltage": (["row"], np.array([3.0, 3.5, 4.0])),
            "current": (["row"], np.array([0.1, 0.2, 0.3])),
            "time": (["row"], np.array([0.0, 1.0, 2.0])),
        }, coords={"row": np.arange(3)})
    
    def test_to_dict(self, sample_dataset):
        """Test to_dict method."""
        raw_data = RawData(data=sample_dataset)
        result = raw_data.to_dict()
        
        assert "voltage" in result
        assert "current" in result
        assert result["voltage"] == [3.0, 3.5, 4.0]
        assert result["current"] == [0.1, 0.2, 0.3]
    
    def test_get_variables(self, sample_dataset):
        """Test get_variables method."""
        raw_data = RawData(data=sample_dataset)
        variables = raw_data.get_variables()
        
        assert len(variables) == 3
        assert "voltage" in variables
        assert "current" in variables
        assert "time" in variables
    
    def test_get_coords(self, sample_dataset):
        """Test get_coords method."""
        raw_data = RawData(data=sample_dataset)
        coords = raw_data.get_coords()
        
        assert "row" in coords
    
    def test_select(self, sample_dataset):
        """Test select method."""
        raw_data = RawData(data=sample_dataset)
        
        # Select specific variables
        selected = raw_data.select(["voltage", "current"])
        assert "voltage" in selected.data_vars
        assert "current" in selected.data_vars
        assert "time" not in selected.data_vars
        
        # Select all (None)
        all_vars = raw_data.select(None)
        assert len(all_vars.data_vars) == 3
    
    def test_getitem(self, sample_dataset):
        """Test __getitem__ method."""
        raw_data = RawData(data=sample_dataset)
        voltage = raw_data["voltage"]
        
        assert isinstance(voltage, xr.DataArray)
        assert len(voltage) == 3
        assert voltage.values[0] == 3.0


class TestMeasurementMetadata:
    """Tests for MeasurementMetadata helper methods."""
    
    def test_to_dict(self):
        """Test to_dict method."""
        info = MeasurementMetadata(
            technique="CV",
            sample_name="Sample1",
            instrument="Potentiostat",
            others={"custom": "value"}
        )
        result = info.to_dict()
        
        assert result["technique"] == "CV"
        assert result["sample_name"] == "Sample1"
        assert result["instrument"] == "Potentiostat"
        assert result["others"] == {"custom": "value"}
    
    def test_get_standard_field(self):
        """Test get method with standard fields."""
        info = MeasurementMetadata(technique="CV", sample_name="Sample1")
        
        assert info.get("technique") == "CV"
        assert info.get("sample_name") == "Sample1"
    
    def test_get_others_field(self):
        """Test get method with others dictionary."""
        info = MeasurementMetadata(
            technique="CV",
            others={"custom_field": "custom_value"}
        )
        
        assert info.get("custom_field") == "custom_value"
    
    def test_get_with_default(self):
        """Test get method with default value."""
        info = MeasurementMetadata(technique="CV")
        
        assert info.get("missing_field", "default") == "default"
    
    def test_update_others(self):
        """Test update_others method."""
        info = MeasurementMetadata(
            technique="CV",
            others={"field1": "value1"}
        )
        
        info.update_others({"field2": "value2", "field3": 123})
        
        assert info.others == {"field1": "value1", "field2": "value2", "field3": 123}


class TestMeasurement:
    """Tests for Measurement helper methods."""
    
    @pytest.fixture
    def sample_measurement(self):
        """Create a sample Measurement for testing."""
        dataset = xr.Dataset({
            "Time/s": (["row"], np.array([0.0, 1.0, 2.0])),
            "Ewe/V": (["row"], np.array([3.0, 3.5, 4.0])),
            "Current/mA": (["row"], np.array([0.1, 0.2, 0.3])),
        }, coords={"row": np.arange(3)})
        
        return Measurement(data=dataset)
    
    def test_to_dict(self, sample_measurement):
        """Test to_dict method."""
        result = sample_measurement.to_dict()
        
        assert "Time/s" in result
        assert "Ewe/V" in result
        assert result["Time/s"] == [0.0, 1.0, 2.0]
    
    def test_get_variables(self, sample_measurement):
        """Test get_variables method."""
        variables = sample_measurement.get_variables()
        
        assert len(variables) == 3
        assert "Time/s" in variables
        assert "Ewe/V" in variables
        assert "Current/mA" in variables
    
    def test_getitem(self, sample_measurement):
        """Test __getitem__ method."""
        time = sample_measurement["Time/s"]
        
        assert isinstance(time, xr.DataArray)
        assert len(time) == 3


class TestAnalysisResultInfo:
    """Tests for AnalysisResultInfo helper methods."""
    
    def test_to_dict(self):
        """Test to_dict method."""
        info = AnalysisResultInfo(
            parameters={"threshold": 0.5, "method": "peak_detection"},
            remarks="Initial analysis"
        )
        result = info.to_dict()
        
        assert result["parameters"] == {"threshold": 0.5, "method": "peak_detection"}
        assert result["remarks"] == "Initial analysis"
    
    def test_get_parameter(self):
        """Test get_parameter method."""
        info = AnalysisResultInfo(parameters={"threshold": 0.5})
        
        assert info.get_parameter("threshold") == 0.5
        assert info.get_parameter("missing") is None
        assert info.get_parameter("missing", 1.0) == 1.0
    
    def test_update_parameters(self):
        """Test update_parameters method."""
        info = AnalysisResultInfo(parameters={"param1": 1.0})
        
        info.update_parameters({"param2": 2.0, "param3": "value"})
        
        assert info.parameters == {"param1": 1.0, "param2": 2.0, "param3": "value"}
    
    def test_add_remark(self):
        """Test add_remark method."""
        info = AnalysisResultInfo(remarks="First remark")
        
        info.add_remark("Second remark")
        
        assert "First remark" in info.remarks
        assert "Second remark" in info.remarks
        assert info.remarks == "First remark\nSecond remark"
    
    def test_add_remark_custom_separator(self):
        """Test add_remark with custom separator."""
        info = AnalysisResultInfo(remarks="First")
        
        info.add_remark("Second", separator=" | ")
        
        assert info.remarks == "First | Second"
    
    def test_add_remark_empty_initial(self):
        """Test add_remark when initial remarks are empty."""
        info = AnalysisResultInfo()
        
        info.add_remark("First remark")
        
        assert info.remarks == "First remark"


class TestAnalysisResult:
    """Tests for AnalysisResult helper methods."""
    
    @pytest.fixture
    def sample_result(self):
        """Create a sample AnalysisResult for testing."""
        dataset = xr.Dataset({
            "peak_voltage": (["peak"], np.array([3.2, 3.8])),
            "peak_current": (["peak"], np.array([0.15, 0.25])),
        }, coords={"peak": np.arange(2)})
        
        return AnalysisResult(data=dataset)
    
    def test_to_dict(self, sample_result):
        """Test to_dict method."""
        result = sample_result.to_dict()
        
        assert "peak_voltage" in result
        assert "peak_current" in result
        assert result["peak_voltage"] == [3.2, 3.8]
    
    def test_get_variables(self, sample_result):
        """Test get_variables method."""
        variables = sample_result.get_variables()
        
        assert len(variables) == 2
        assert "peak_voltage" in variables
        assert "peak_current" in variables


class TestNXField:
    """Tests for NXField helper methods."""
    
    def test_from_dataarray_basic(self):
        """Test from_dataarray with basic DataArray."""
        da = xr.DataArray(
            3.14,
            attrs={"units": "V", "type": "NX_FLOAT", "EX_doc": "Test field"}
        )
        
        field = NXField.from_dataarray("test_field", da)
        
        assert field.name == "test_field"
        assert field.value == 3.14
        assert field.units == "V"
        assert field.dtype == "NX_FLOAT"
        assert field.doc == "Test field"
    
    def test_from_dataarray_with_dtype_override(self):
        """Test from_dataarray with dtype override."""
        da = xr.DataArray(42, attrs={"type": "NX_FLOAT"})
        
        field = NXField.from_dataarray("test", da, dtype="NX_INT")
        
        assert field.dtype == "NX_INT"
    
    def test_from_dataarray_array_value(self):
        """Test from_dataarray with array value."""
        da = xr.DataArray(np.array([1.0, 2.0, 3.0]))
        
        field = NXField.from_dataarray("array_field", da)
        
        assert isinstance(field.value, np.ndarray)
        assert len(field.value) == 3


class TestNXGroup:
    """Tests for NXGroup helper methods."""
    
    def test_add_field(self):
        """Test add_field fluent interface."""
        group = NXGroup(name="test_group", nx_class="NXcollection")
        field1 = NXField("field1", 1.0, "NX_FLOAT")
        field2 = NXField("field2", 2.0, "NX_FLOAT")
        
        result = group.add_field(field1).add_field(field2)
        
        assert result is group  # Fluent interface
        assert len(group.fields) == 2
        assert group.fields[0].name == "field1"
        assert group.fields[1].name == "field2"
    
    def test_add_group(self):
        """Test add_group fluent interface."""
        parent = NXGroup(name="parent", nx_class="NXentry")
        child1 = NXGroup(name="child1", nx_class="NXdata")
        child2 = NXGroup(name="child2", nx_class="NXdata")
        
        result = parent.add_group(child1).add_group(child2)
        
        assert result is parent
        assert len(parent.groups) == 2
        assert parent.groups[0].name == "child1"
    
    def test_from_dataset(self):
        """Test from_dataset class method."""
        dataset = xr.Dataset({
            "var1": xr.DataArray(1.0, attrs={"units": "V"}),
            "var2": xr.DataArray(2.0, attrs={"units": "A"}),
        }, attrs={"EX_doc": "Test dataset", "NX_class": "NXdata"})
        
        group = NXGroup.from_dataset("test_group", dataset, nx_class="NXcollection")
        
        assert group.name == "test_group"
        assert group.nx_class == "NXcollection"
        assert len(group.fields) == 2
        assert group.doc == "Test dataset"
        
        # Check fields were created
        field_names = [f.name for f in group.fields]
        assert "var1" in field_names
        assert "var2" in field_names


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""
    
    def test_rawmetadata_alias(self):
        """Test RawMetadata is alias for RawDataInfo."""
        assert RawMetadata is RawDataInfo
    
    def test_measurementinfo_alias(self):
        """Test MeasurementInfo is alias for MeasurementMetadata."""
        assert MeasurementInfo is MeasurementMetadata
    
    def test_results_alias(self):
        """Test Results is alias for AnalysisResult."""
        assert Results is AnalysisResult
    
    def test_resultsinfo_alias(self):
        """Test ResultsInfo is alias for AnalysisResultInfo."""
        assert ResultsInfo is AnalysisResultInfo
    
    def test_can_use_old_names(self):
        """Test that old names still work."""
        # Should be able to create instances with old names
        info = RawMetadata(meta={"test": "value"})
        assert isinstance(info, RawDataInfo)
        
        meas_info = MeasurementInfo(technique="CV")
        assert isinstance(meas_info, MeasurementMetadata)
        
        results = Results(data=xr.Dataset())
        assert isinstance(results, AnalysisResult)
        
        results_info = ResultsInfo(parameters={})
        assert isinstance(results_info, AnalysisResultInfo)
