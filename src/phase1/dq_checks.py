"""
Data quality checks and flagging for Phase 1 data.

This module provides functions to perform data quality checks,
add validation flags, and identify problematic rows.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from .id_normalizer import is_valid_asin, is_valid_vendor_id, is_valid_week

logger = logging.getLogger(__name__)


def check_missing_required_columns(
    df: pd.DataFrame, required_columns: List[str]
) -> pd.Series:
    """
    Flag rows with missing values in required columns.

    Args:
        df: DataFrame to check
        required_columns: List of required column names

    Returns:
        Boolean series (True = has missing required values)
    """
    missing_mask = pd.Series([False] * len(df), index=df.index)

    for col in required_columns:
        if col in df.columns:
            missing_mask |= df[col].isna()

    return missing_mask


def check_invalid_ids(df: pd.DataFrame) -> pd.Series:
    """
    Flag rows with invalid ID formats (ASIN, vendor_id, week).

    Args:
        df: DataFrame to check (must have normalized IDs)

    Returns:
        Boolean series (True = has invalid IDs)
    """
    invalid_mask = pd.Series([False] * len(df), index=df.index)

    # Check ASIN format
    if "asin" in df.columns:
        asin_valid = df["asin"].apply(
            lambda x: is_valid_asin(str(x)) if pd.notna(x) else False
        )
        invalid_mask |= ~asin_valid

    # Check vendor_id format
    if "vendor_id" in df.columns:
        vendor_valid = df["vendor_id"].apply(
            lambda x: is_valid_vendor_id(str(x)) if pd.notna(x) else False
        )
        invalid_mask |= ~vendor_valid

    # Check week format
    if "week" in df.columns:
        week_valid = df["week"].apply(
            lambda x: (
                is_valid_week(int(x))
                if pd.notna(x) and isinstance(x, (int, np.integer))
                else False
            )
        )
        invalid_mask |= ~week_valid

    return invalid_mask


def check_invalid_numeric(df: pd.DataFrame, numeric_columns: List[str]) -> pd.Series:
    """
    Flag rows with invalid numeric values (NaN, inf, etc.).

    Args:
        df: DataFrame to check
        numeric_columns: List of columns that should be numeric

    Returns:
        Boolean series (True = has invalid numeric values)
    """
    invalid_mask = pd.Series([False] * len(df), index=df.index)

    for col in numeric_columns:
        if col in df.columns:
            # Check for NaN or infinite values
            invalid_mask |= df[col].isna() | np.isinf(df[col])

    return invalid_mask


def check_out_of_range(df: pd.DataFrame, non_negative_fields: List[str]) -> pd.Series:
    """
    Flag rows with out-of-range numeric values.

    Args:
        df: DataFrame to check
        non_negative_fields: List of columns that should not be negative

    Returns:
        Boolean series (True = has out-of-range values)
    """
    out_of_range_mask = pd.Series([False] * len(df), index=df.index)

    for col in non_negative_fields:
        if col in df.columns:
            # Check for negative values
            out_of_range_mask |= (df[col] < 0) & df[col].notna()

    return out_of_range_mask


def add_data_quality_flags(
    df: pd.DataFrame,
    required_columns: List[str],
    numeric_columns: List[str],
    non_negative_fields: List[str],
) -> pd.DataFrame:
    """
    Add data quality flag columns to DataFrame.

    Adds the following boolean columns:
    - dq_missing_required_columns: Missing values in required fields
    - dq_invalid_ids: Invalid ID formats
    - dq_invalid_numeric: Invalid numeric values
    - dq_out_of_range: Out-of-range values
    - is_valid_row: Aggregate flag (False if any DQ issue)

    Args:
        df: DataFrame to add flags to
        required_columns: List of required column names
        numeric_columns: List of numeric column names
        non_negative_fields: List of non-negative column names

    Returns:
        DataFrame with added DQ flag columns
    """
    df = df.copy()

    logger.info("Adding data quality flags...")

    # Add individual DQ flags
    df["dq_missing_required_columns"] = check_missing_required_columns(
        df, required_columns
    )
    df["dq_invalid_ids"] = check_invalid_ids(df)
    df["dq_invalid_numeric"] = check_invalid_numeric(df, numeric_columns)
    df["dq_out_of_range"] = check_out_of_range(df, non_negative_fields)

    # Aggregate flag: row is valid if no DQ issues
    df["is_valid_row"] = ~(
        df["dq_missing_required_columns"]
        | df["dq_invalid_ids"]
        | df["dq_invalid_numeric"]
        | df["dq_out_of_range"]
    )

    # Log DQ summary
    total_rows = len(df)
    invalid_rows = (~df["is_valid_row"]).sum()
    missing_required = df["dq_missing_required_columns"].sum()
    invalid_ids = df["dq_invalid_ids"].sum()
    invalid_numeric = df["dq_invalid_numeric"].sum()
    out_of_range = df["dq_out_of_range"].sum()

    logger.info("Data Quality Summary:")
    logger.info(f"  Total rows: {total_rows}")
    logger.info(
        f"  Valid rows: {total_rows - invalid_rows} "
        f"({100 * (total_rows - invalid_rows) / total_rows:.1f}%)"
    )
    logger.info(
        f"  Invalid rows: {invalid_rows} ({100 * invalid_rows / total_rows:.1f}%)"
    )
    logger.info(f"    - Missing required columns: {missing_required}")
    logger.info(f"    - Invalid IDs: {invalid_ids}")
    logger.info(f"    - Invalid numeric: {invalid_numeric}")
    logger.info(f"    - Out of range: {out_of_range}")

    return df


def get_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rows flagged as invalid.

    Args:
        df: DataFrame with is_valid_row column

    Returns:
        DataFrame containing only invalid rows
    """
    if "is_valid_row" not in df.columns:
        logger.warning("DataFrame does not have is_valid_row column")
        return pd.DataFrame()

    invalid = df[~df["is_valid_row"]].copy()
    logger.info(f"Extracted {len(invalid)} invalid rows")

    return invalid


def calculate_null_percentages(df: pd.DataFrame) -> dict:
    """
    Calculate null percentage for each column.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary mapping column names to null percentages
    """
    null_pct = {}
    total_rows = len(df)

    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct[col] = (
            round(100 * null_count / total_rows, 2) if total_rows > 0 else 0.0
        )

    return null_pct


def detect_numeric_outliers(
    df: pd.DataFrame, numeric_columns: List[str], threshold: float = 3.0
) -> dict:
    """
    Detect outliers in numeric columns using z-score method.

    Args:
        df: DataFrame to analyze
        numeric_columns: List of numeric column names
        threshold: Z-score threshold for outlier detection (default: 3.0)

    Returns:
        Dictionary mapping column names to outlier counts
    """
    outlier_counts = {}

    for col in numeric_columns:
        if col not in df.columns:
            continue

        # Calculate z-scores
        data = df[col].dropna()
        if len(data) == 0:
            outlier_counts[col] = 0
            continue

        mean = data.mean()
        std = data.std()

        if std == 0:
            outlier_counts[col] = 0
            continue

        z_scores = np.abs((data - mean) / std)
        outliers = (z_scores > threshold).sum()
        outlier_counts[col] = int(outliers)

    return outlier_counts
