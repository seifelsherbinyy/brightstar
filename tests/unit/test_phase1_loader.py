"""
Unit tests for Phase 1 loader system.

Tests cover schema validation, ID normalization, data quality checks,
and end-to-end loading functionality.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from phase1.dq_checks import (
    add_data_quality_flags,
    calculate_null_percentages,
    check_invalid_ids,
    check_missing_required_columns,
    check_out_of_range,
    detect_numeric_outliers,
)
from phase1.id_normalizer import (
    is_valid_asin,
    is_valid_vendor_id,
    is_valid_week,
    normalize_asin,
    normalize_category,
    normalize_vendor_id,
    normalize_week,
)
from phase1.loader import (
    cast_to_schema_types,
    detect_file_format,
    load_phase1,
    normalize_column_names,
    normalize_ids,
)
from phase1.schema_validator import SchemaValidator
from phase1.validation_report import generate_validation_report


class TestIDNormalizer:
    """Test ID normalization functions."""

    def test_normalize_asin_basic(self):
        """Test basic ASIN normalization."""
        assert normalize_asin("B07XYZ1234") == "B07XYZ1234"
        assert normalize_asin("b07xyz1234") == "B07XYZ1234"
        assert normalize_asin("B07-XYZ-1234") == "B07XYZ1234"
        assert normalize_asin("  B07XYZ1234  ") == "B07XYZ1234"

    def test_normalize_asin_padding(self):
        """Test ASIN padding to 10 characters."""
        assert normalize_asin("B07XYZ") == "0000B07XYZ"
        assert normalize_asin("123") == "0000000123"

    def test_normalize_asin_invalid(self):
        """Test ASIN normalization with invalid input."""
        with pytest.raises(ValueError):
            normalize_asin(None)
        with pytest.raises(ValueError):
            normalize_asin("")
        with pytest.raises(ValueError):
            normalize_asin("nan")

    def test_normalize_vendor_id_basic(self):
        """Test basic vendor ID normalization."""
        assert normalize_vendor_id("V_001") == "V_001"
        assert normalize_vendor_id("v_001") == "V_001"
        assert normalize_vendor_id("V-001") == "V001"
        assert normalize_vendor_id("  V_001  ") == "V_001"

    def test_normalize_vendor_id_invalid(self):
        """Test vendor ID normalization with invalid input."""
        with pytest.raises(ValueError):
            normalize_vendor_id(None)
        with pytest.raises(ValueError):
            normalize_vendor_id("")

    def test_normalize_week_basic(self):
        """Test basic week normalization."""
        assert normalize_week(202501) == 202501
        assert normalize_week("202501") == 202501
        assert normalize_week(202553) == 202553

    def test_normalize_week_invalid(self):
        """Test week normalization with invalid input."""
        with pytest.raises(ValueError):
            normalize_week(None)
        with pytest.raises(ValueError):
            normalize_week("")
        with pytest.raises(ValueError):
            normalize_week(20251)  # Too short
        with pytest.raises(ValueError):
            normalize_week(202554)  # Week > 53
        with pytest.raises(ValueError):
            normalize_week(201901)  # Year < 2020

    def test_normalize_category_basic(self):
        """Test basic category normalization."""
        assert normalize_category("frozen food") == "Frozen Food"
        assert normalize_category("FROZEN FOOD") == "Frozen Food"
        assert normalize_category("  frozen  food  ") == "Frozen Food"

    def test_is_valid_asin(self):
        """Test ASIN validation."""
        assert is_valid_asin("B07XYZ1234") is True
        assert is_valid_asin("0000B07XYZ") is True
        assert is_valid_asin("B07XYZ123") is False  # Too short
        assert is_valid_asin("b07xyz1234") is False  # Not uppercase
        assert is_valid_asin("") is False

    def test_is_valid_vendor_id(self):
        """Test vendor ID validation."""
        assert is_valid_vendor_id("V_001") is True
        assert is_valid_vendor_id("VENDOR123") is True
        assert is_valid_vendor_id("v_001") is False  # Not uppercase
        assert is_valid_vendor_id("") is False

    def test_is_valid_week(self):
        """Test week validation."""
        assert is_valid_week(202501) is True
        assert is_valid_week(209952) is True
        assert is_valid_week(202554) is False  # Week > 53
        assert is_valid_week(201901) is False  # Year < 2020


class TestSchemaValidator:
    """Test schema validation functionality."""

    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create a temporary schema file."""
        schema = {
            "required_columns": ["vendor_id", "asin", "week", "gms"],
            "dtypes": {
                "vendor_id": "string",
                "asin": "string",
                "week": "int",
                "gms": "float",
            },
            "categorical_fields": {"category": ["Frozen Food", "Baking Supplies"]},
            "validation": {
                "non_negative_fields": ["gms"],
            },
        }
        schema_path = tmp_path / "test_schema.yaml"
        with open(schema_path, "w") as f:
            yaml.dump(schema, f)
        return schema_path

    def test_schema_validator_init(self, schema_file):
        """Test schema validator initialization."""
        validator = SchemaValidator(str(schema_file))
        assert validator.schema is not None
        assert "required_columns" in validator.schema

    def test_schema_validator_missing_file(self):
        """Test schema validator with missing file."""
        with pytest.raises(FileNotFoundError):
            SchemaValidator("nonexistent.yaml")

    def test_validate_required_columns(self, schema_file):
        """Test required column validation."""
        validator = SchemaValidator(str(schema_file))
        df = pd.DataFrame(
            {
                "vendor_id": ["V1"],
                "asin": ["B07XYZ1234"],
                "week": [202501],
            }
        )
        missing = validator.validate_required_columns(df)
        assert "gms" in missing

    def test_validate_categorical(self, schema_file):
        """Test categorical validation."""
        validator = SchemaValidator(str(schema_file))
        df = pd.DataFrame({"category": ["Frozen Food", "Invalid", "Baking Supplies"]})
        is_valid = validator.validate_categorical(df, "category")
        assert is_valid.iloc[0]
        assert not is_valid.iloc[1]
        assert is_valid.iloc[2]


class TestDQChecks:
    """Test data quality check functions."""

    def test_check_missing_required_columns(self):
        """Test missing required columns check."""
        df = pd.DataFrame(
            {
                "vendor_id": ["V1", None, "V3"],
                "asin": ["A1", "A2", "A3"],
            }
        )
        missing = check_missing_required_columns(df, ["vendor_id", "asin"])
        assert not missing.iloc[0]
        assert missing.iloc[1]
        assert not missing.iloc[2]

    def test_check_invalid_ids(self):
        """Test invalid ID check."""
        df = pd.DataFrame(
            {
                "asin": ["B07XYZ1234", "invalid", "0000123456"],
                "vendor_id": ["V_001", "V_002", "V_003"],
                "week": [202501, 202502, 999999],
            }
        )
        invalid = check_invalid_ids(df)
        assert not invalid.iloc[0]
        assert invalid.iloc[1]  # Invalid ASIN
        assert invalid.iloc[2]  # Invalid week

    def test_check_out_of_range(self):
        """Test out of range check."""
        df = pd.DataFrame(
            {
                "gms": [100.0, -50.0, 200.0],
                "asp": [10.0, 5.0, -1.0],
            }
        )
        out_of_range = check_out_of_range(df, ["gms", "asp"])
        assert not out_of_range.iloc[0]
        assert out_of_range.iloc[1]  # Negative GMS
        assert out_of_range.iloc[2]  # Negative ASP

    def test_add_data_quality_flags(self):
        """Test adding DQ flags to DataFrame."""
        df = pd.DataFrame(
            {
                "vendor_id": ["V_001", "V_002", None],
                "asin": ["B07XYZ1234", "invalid", "0000123456"],
                "week": [202501, 202502, 202503],
                "gms": [100.0, -50.0, 200.0],
            }
        )
        df_flagged = add_data_quality_flags(
            df,
            required_columns=["vendor_id", "asin", "week", "gms"],
            numeric_columns=["gms"],
            non_negative_fields=["gms"],
        )
        assert "is_valid_row" in df_flagged.columns
        assert "dq_missing_required_columns" in df_flagged.columns
        assert "dq_invalid_ids" in df_flagged.columns
        assert "dq_out_of_range" in df_flagged.columns

    def test_calculate_null_percentages(self):
        """Test null percentage calculation."""
        df = pd.DataFrame(
            {
                "col1": [1, None, 3, None],
                "col2": [1, 2, 3, 4],
            }
        )
        null_pct = calculate_null_percentages(df)
        assert null_pct["col1"] == 50.0
        assert null_pct["col2"] == 0.0

    def test_detect_numeric_outliers(self):
        """Test outlier detection."""
        df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            }
        )
        outliers = detect_numeric_outliers(df, ["value"], threshold=2.0)
        assert outliers["value"] > 0


class TestLoader:
    """Test main loader functionality."""

    def test_detect_file_format(self):
        """Test file format detection."""
        assert detect_file_format("data.csv") == "csv"
        assert detect_file_format("data.parquet") == "parquet"
        assert detect_file_format("data.pq") == "parquet"
        with pytest.raises(ValueError):
            detect_file_format("data.txt")

    def test_normalize_column_names(self):
        """Test column name normalization."""
        df = pd.DataFrame(
            {
                "Vendor ID": [1],
                "ASIN ": [2],
                "Week Label": [3],
            }
        )
        df_norm = normalize_column_names(df)
        assert "vendor_id" in df_norm.columns
        assert "asin" in df_norm.columns
        assert "week_label" in df_norm.columns

    def test_cast_to_schema_types(self):
        """Test type casting."""
        df = pd.DataFrame(
            {
                "vendor_id": [1, 2, 3],
                "week": ["202501", "202502", "202503"],
                "gms": ["100", "200.5", "300"],
            }
        )
        dtype_map = {
            "vendor_id": "string",
            "week": "int",
            "gms": "float",
        }
        df_cast = cast_to_schema_types(df, dtype_map)
        assert df_cast["vendor_id"].dtype == object
        assert df_cast["week"].dtype == "int64"
        assert df_cast["gms"].dtype == "float64"

    def test_normalize_ids_dataframe(self):
        """Test ID normalization on DataFrame."""
        df = pd.DataFrame(
            {
                "vendor_id": ["v_001", "v_002"],
                "asin": ["b07xyz1234", "b08abc5678"],
                "week": [202501, 202502],
                "category": ["frozen food", "baking supplies"],
            }
        )
        df_norm = normalize_ids(df)
        assert df_norm["vendor_id"].iloc[0] == "V_001"
        assert df_norm["asin"].iloc[0] == "B07XYZ1234"
        assert df_norm["week"].iloc[0] == 202501
        assert df_norm["category"].iloc[0] == "Frozen Food"


class TestEndToEnd:
    """Test end-to-end loader functionality."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample CSV data file."""
        data = {
            "Vendor_ID": ["V_001", "V_002", "V_003"],
            "ASIN": ["B07XYZ1234", "B08ABC5678", "B09DEF9012"],
            "Week": [202501, 202502, 202503],
            "Category": ["Frozen Food", "Baking Supplies", "Health Care"],
            "Subcategory": ["Ice Cream", "Flour", "Vitamins"],
            "GMS": [1000.0, 2000.0, 1500.0],
            "Ordered_Units": [100, 200, 150],
            "Shipped_Units": [95, 190, 145],
            "ASP": [10.0, 10.0, 10.0],
            "CP": [5.0, 5.0, 5.0],
            "CPPU": [0.05, 0.025, 0.033],
            "GV": [500.0, 1000.0, 750.0],
            "Net_PPM": [0.0, 0.0, 0.0],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "sample.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def schema_file(self, tmp_path):
        """Create schema file for testing."""
        schema_path = Path(__file__).parent.parent / "config" / "schema_phase1.yaml"
        if schema_path.exists():
            return str(schema_path)

        # Create a test schema if config doesn't exist
        schema = {
            "required_columns": [
                "vendor_id",
                "asin",
                "week",
                "category",
                "subcategory",
                "gms",
                "ordered_units",
                "shipped_units",
                "asp",
                "cp",
                "cppu",
                "gv",
                "net_ppm",
            ],
            "dtypes": {
                "vendor_id": "string",
                "asin": "string",
                "week": "int",
                "category": "string",
                "subcategory": "string",
                "gms": "float",
                "ordered_units": "float",
                "shipped_units": "float",
                "asp": "float",
                "cp": "float",
                "cppu": "float",
                "gv": "float",
                "net_ppm": "float",
            },
            "categorical_fields": {
                "category": ["Frozen Food", "Baking Supplies", "Health Care"]
            },
            "validation": {
                "non_negative_fields": ["gms", "ordered_units", "shipped_units", "asp"],
            },
        }
        test_schema_path = tmp_path / "test_schema.yaml"
        with open(test_schema_path, "w") as f:
            yaml.dump(schema, f)
        return str(test_schema_path)

    def test_load_phase1_end_to_end(self, sample_data, schema_file, tmp_path):
        """Test complete load_phase1 workflow."""
        output_dir = tmp_path / "output"

        df = load_phase1(
            path=str(sample_data),
            schema_path=schema_file,
            output_dir=str(output_dir),
            log_level="INFO",
        )

        # Verify output
        assert df is not None
        assert len(df) == 3
        assert "is_valid_row" in df.columns
        assert "dq_missing_required_columns" in df.columns

        # Verify all rows are valid
        assert df["is_valid_row"].all()

        # Verify normalized IDs
        assert df["vendor_id"].iloc[0] == "V_001"
        assert df["asin"].iloc[0] == "B07XYZ1234"
        assert df["category"].iloc[0] == "Frozen Food"

        # Verify output files
        assert (output_dir / "phase1_validated.parquet").exists()
        assert (output_dir / "phase1_validation_report.json").exists()

        # Verify validation report
        with open(output_dir / "phase1_validation_report.json") as f:
            report = json.load(f)
        assert report["row_count"] == 3
        assert report["valid_row_count"] == 3
        assert report["invalid_row_count"] == 0


class TestValidationReport:
    """Test validation report generation."""

    def test_generate_validation_report(self, tmp_path):
        """Test validation report generation."""
        df = pd.DataFrame(
            {
                "vendor_id": ["V_001", "V_002"],
                "asin": ["B07XYZ1234", "B08ABC5678"],
                "gms": [1000.0, 2000.0],
                "is_valid_row": [True, True],
                "dq_missing_required_columns": [False, False],
                "dq_invalid_ids": [False, False],
                "dq_invalid_numeric": [False, False],
                "dq_out_of_range": [False, False],
            }
        )

        report_path = tmp_path / "report.json"
        report = generate_validation_report(
            df,
            numeric_columns=["gms"],
            output_path=str(report_path),
        )

        assert report is not None
        assert report["row_count"] == 2
        assert report["valid_row_count"] == 2
        assert report_path.exists()

        # Verify JSON structure
        with open(report_path) as f:
            saved_report = json.load(f)
        assert saved_report["row_count"] == 2
        assert "timestamp" in saved_report
        assert "null_percentage_by_column" in saved_report
