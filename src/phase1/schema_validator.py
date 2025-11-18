"""
Schema validation for Phase 1 data against YAML configuration.

This module provides validation logic to ensure data conforms to the
canonical schema defined in schema_phase1.yaml.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class SchemaMismatchError(Exception):
    """Raised when data does not match expected schema."""

    pass


class SchemaValidator:
    """Validates DataFrames against YAML schema definition."""

    def __init__(self, schema_path: str):
        """
        Initialize schema validator.

        Args:
            schema_path: Path to YAML schema file

        Raises:
            FileNotFoundError: If schema file not found
            yaml.YAMLError: If schema file is invalid
        """
        self.schema_path = Path(schema_path)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(self.schema_path, "r") as f:
            self.schema = yaml.safe_load(f)

        logger.info(f"Loaded schema from {schema_path}")

    def validate_required_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Validate that all required columns are present.

        Args:
            df: DataFrame to validate

        Returns:
            List of missing column names (empty if all present)
        """
        required = self.schema.get("required_columns", [])
        existing = set(df.columns)
        missing = [col for col in required if col not in existing]

        if missing:
            logger.error(f"Missing required columns: {missing}")
        else:
            logger.info("All required columns present")

        return missing

    def get_dtype_map(self) -> Dict[str, str]:
        """
        Get column data type mapping from schema.

        Returns:
            Dictionary mapping column names to expected data types
        """
        return self.schema.get("dtypes", {})

    def get_categorical_values(self, column: str) -> Optional[List[str]]:
        """
        Get allowed categorical values for a column.

        Args:
            column: Column name

        Returns:
            List of allowed values, or None if not a categorical field
        """
        categorical = self.schema.get("categorical_fields", {})
        return categorical.get(column)

    def validate_categorical(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Validate categorical column values against schema.

        Args:
            df: DataFrame containing the column
            column: Column name to validate

        Returns:
            Boolean series indicating valid rows (True = valid)
        """
        allowed_values = self.get_categorical_values(column)
        if allowed_values is None or column not in df.columns:
            return pd.Series([True] * len(df), index=df.index)

        is_valid = df[column].isin(allowed_values) | df[column].isna()
        invalid_count = (~is_valid).sum()

        if invalid_count > 0:
            logger.warning(
                f"Column '{column}' has {invalid_count} invalid categorical values"
            )

        return is_valid

    def get_validation_rules(self) -> Dict[str, Any]:
        """
        Get validation rules from schema.

        Returns:
            Dictionary of validation rules
        """
        return self.schema.get("validation", {})

    def get_non_negative_fields(self) -> List[str]:
        """
        Get list of fields that should not be negative.

        Returns:
            List of column names
        """
        validation = self.get_validation_rules()
        return validation.get("non_negative_fields", [])

    def get_week_format_rules(self) -> Dict[str, Any]:
        """
        Get week format validation rules.

        Returns:
            Dictionary with week format rules
        """
        validation = self.get_validation_rules()
        return validation.get("week_format", {})

    def get_asin_format_rules(self) -> Dict[str, Any]:
        """
        Get ASIN format validation rules.

        Returns:
            Dictionary with ASIN format rules
        """
        validation = self.get_validation_rules()
        return validation.get("asin_format", {})

    def get_vendor_id_format_rules(self) -> Dict[str, Any]:
        """
        Get vendor ID format validation rules.

        Returns:
            Dictionary with vendor ID format rules
        """
        validation = self.get_validation_rules()
        return validation.get("vendor_id_format", {})

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete schema validation.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results

        Raises:
            SchemaMismatchError: If required columns are missing
        """
        results = {
            "valid": True,
            "missing_columns": [],
            "dtype_mismatches": [],
            "categorical_violations": {},
        }

        # Check required columns
        missing = self.validate_required_columns(df)
        if missing:
            results["missing_columns"] = missing
            results["valid"] = False
            raise SchemaMismatchError(f"Missing required columns: {missing}")

        # Check data types
        dtype_map = self.get_dtype_map()
        for col, expected_type in dtype_map.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                # Map pandas dtypes to schema types
                if expected_type == "string" and not actual_type.startswith(
                    ("object", "string")
                ):
                    results["dtype_mismatches"].append(
                        {
                            "column": col,
                            "expected": expected_type,
                            "actual": actual_type,
                        }
                    )
                elif expected_type == "int" and not actual_type.startswith("int"):
                    results["dtype_mismatches"].append(
                        {
                            "column": col,
                            "expected": expected_type,
                            "actual": actual_type,
                        }
                    )
                elif expected_type == "float" and not actual_type.startswith(
                    ("float", "int")
                ):
                    results["dtype_mismatches"].append(
                        {
                            "column": col,
                            "expected": expected_type,
                            "actual": actual_type,
                        }
                    )

        # Check categorical values
        categorical_fields = self.schema.get("categorical_fields", {})
        for col in categorical_fields:
            if col in df.columns:
                is_valid = self.validate_categorical(df, col)
                invalid_count = (~is_valid).sum()
                if invalid_count > 0:
                    results["categorical_violations"][col] = invalid_count
                    results["valid"] = False

        if results["dtype_mismatches"]:
            logger.warning(f"Found {len(results['dtype_mismatches'])} dtype mismatches")

        return results
