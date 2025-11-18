"""
Main loader module for Phase 1 data ingestion.

This module provides the load_phase1() function that orchestrates
schema validation, normalization, and output generation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .dq_checks import add_data_quality_flags
from .id_normalizer import (
    normalize_asin,
    normalize_category,
    normalize_vendor_id,
    normalize_week,
)
from .schema_validator import SchemaValidator
from .validation_report import (
    generate_validation_report,
    print_validation_summary,
    write_invalid_rows_report,
)

logger = logging.getLogger(__name__)


class IDFormatError(Exception):
    """Raised when ID normalization fails."""

    pass


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for Phase 1 loader.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def detect_file_format(path: str) -> str:
    """
    Auto-detect file format from extension.

    Args:
        path: File path

    Returns:
        File format ("csv" or "parquet")

    Raises:
        ValueError: If unsupported file format
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return "csv"
    elif suffix in [".parquet", ".pq"]:
        return "parquet"
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw data from CSV or Parquet file.

    Args:
        path: Path to raw data file

    Returns:
        DataFrame with raw data

    Raises:
        FileNotFoundError: If file not found
        ValueError: If unsupported format
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    file_format = detect_file_format(path)
    logger.info(f"Loading {file_format.upper()} file: {path}")

    if file_format == "csv":
        df = pd.read_csv(path)
    else:  # parquet
        df = pd.read_parquet(path)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase and standard format.

    Args:
        df: DataFrame with raw column names

    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    logger.info("Normalized column names")
    return df


def cast_to_schema_types(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    """
    Cast columns to schema-specified data types with safe coercion.

    Args:
        df: DataFrame to cast
        dtype_map: Dictionary mapping column names to target types

    Returns:
        DataFrame with casted types
    """
    df = df.copy()

    for col, target_type in dtype_map.items():
        if col not in df.columns:
            continue

        try:
            if target_type == "string":
                df[col] = df[col].astype(str).replace("nan", np.nan)
            elif target_type == "int":
                # Convert to numeric first, then int (coerce errors to NaN)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(0).astype(int)
            elif target_type == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Convert percentages (values like "95%") to decimals
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.rstrip("%")
                    df[col] = pd.to_numeric(df[col], errors="coerce") / 100

            logger.debug(f"Cast column '{col}' to {target_type}")
        except Exception as e:
            logger.warning(f"Failed to cast column '{col}' to {target_type}: {e}")

    return df


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ID columns (ASIN, vendor_id, week, category).

    Args:
        df: DataFrame with raw IDs

    Returns:
        DataFrame with normalized IDs
    """
    df = df.copy()

    # Normalize ASINs
    if "asin" in df.columns:
        logger.info("Normalizing ASINs...")
        df["asin"] = df["asin"].apply(
            lambda x: normalize_asin(x) if pd.notna(x) else np.nan
        )

    # Normalize vendor IDs
    if "vendor_id" in df.columns:
        logger.info("Normalizing vendor IDs...")
        df["vendor_id"] = df["vendor_id"].apply(
            lambda x: normalize_vendor_id(x) if pd.notna(x) else np.nan
        )

    # Normalize weeks
    if "week" in df.columns:
        logger.info("Normalizing weeks...")
        df["week"] = df["week"].apply(
            lambda x: normalize_week(x) if pd.notna(x) else np.nan
        )

    # Normalize categories
    if "category" in df.columns:
        logger.info("Normalizing categories...")
        df["category"] = df["category"].apply(
            lambda x: normalize_category(x) if pd.notna(x) else np.nan
        )

    if "subcategory" in df.columns:
        logger.info("Normalizing subcategories...")
        df["subcategory"] = df["subcategory"].apply(
            lambda x: normalize_category(x) if pd.notna(x) else np.nan
        )

    return df


def load_phase1(
    path: str,
    schema_path: str = "config/schema_phase1.yaml",
    output_dir: str = "output/",
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Load and validate Phase 1 data with schema enforcement.

    This function:
    1. Loads raw CSV/Parquet data
    2. Validates against YAML schema
    3. Normalizes IDs and categories
    4. Performs data quality checks
    5. Generates validation report
    6. Writes validated Parquet output

    Args:
        path: Path to raw Phase 1 CSV/Parquet file
        schema_path: Path to YAML schema definition (default: config/schema_phase1.yaml)
        output_dir: Directory for validated outputs (default: output/)
        log_level: Logging level (default: INFO)

    Returns:
        Validated and normalized DataFrame

    Raises:
        FileNotFoundError: If input file or schema not found
        SchemaMismatchError: If data doesn't match schema
        ValueError: If data validation fails
    """
    # Setup logging
    setup_logging(log_level)
    logger.info("=" * 60)
    logger.info("PHASE 1 DATA LOADER - STARTED")
    logger.info("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load schema
    logger.info(f"Loading schema from: {schema_path}")
    validator = SchemaValidator(schema_path)

    # Load raw data
    logger.info(f"Loading raw data from: {path}")
    df = load_raw_data(path)

    if len(df) == 0:
        raise ValueError("Input file is empty (0 rows)")

    # Normalize column names
    df = normalize_column_names(df)

    # Validate schema
    logger.info("Validating schema...")
    _ = validator.validate_schema(df)

    # Cast to schema types
    logger.info("Casting to schema types...")
    dtype_map = validator.get_dtype_map()
    df = cast_to_schema_types(df, dtype_map)

    # Normalize IDs
    logger.info("Normalizing IDs...")
    try:
        df = normalize_ids(df)
    except Exception as e:
        logger.error(f"ID normalization failed: {e}")
        raise IDFormatError(f"Failed to normalize IDs: {e}")

    # Get schema metadata for DQ checks
    required_columns = validator.schema.get("required_columns", [])
    numeric_columns = [
        col for col, dtype in dtype_map.items() if dtype in ["float", "int"]
    ]
    non_negative_fields = validator.get_non_negative_fields()

    # Add data quality flags
    logger.info("Performing data quality checks...")
    df = add_data_quality_flags(
        df, required_columns, numeric_columns, non_negative_fields
    )

    # Generate validation report
    logger.info("Generating validation report...")
    report_path = output_path / "phase1_validation_report.json"
    report = generate_validation_report(
        df,
        numeric_columns,
        str(report_path),
        additional_info={
            "input_file": str(path),
            "schema_file": str(schema_path),
        },
    )

    # Print summary
    print_validation_summary(report)

    # Write invalid rows to CSV
    invalid_rows_path = output_path / "phase1_invalid_rows.csv"
    write_invalid_rows_report(df, str(invalid_rows_path))

    # Write validated parquet
    output_parquet = output_path / "phase1_validated.parquet"
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Validated data written to: {output_parquet}")

    logger.info("=" * 60)
    logger.info("PHASE 1 DATA LOADER - COMPLETED")
    logger.info("=" * 60)

    return df
