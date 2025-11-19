"""
Validation report generation for Phase 1 data.

This module provides functions to generate JSON validation reports
with data quality metrics and summary statistics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .dq_checks import calculate_null_percentages, detect_numeric_outliers

logger = logging.getLogger(__name__)


def generate_validation_report(
    df: pd.DataFrame,
    numeric_columns: List[str],
    output_path: str,
    additional_info: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive validation report for Phase 1 data.

    Args:
        df: Validated DataFrame with DQ flags
        numeric_columns: List of numeric column names
        output_path: Path to write JSON report
        additional_info: Optional additional metadata to include

    Returns:
        Dictionary containing validation report data
    """
    logger.info("Generating validation report...")

    report = {
        "timestamp": datetime.now().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "invalid_row_count": 0,
        "invalid_id_rows": 0,
        "missing_required_columns": [],
        "null_percentage_by_column": {},
        "numeric_outlier_counts": {},
        "data_quality_summary": {},
    }

    # Count invalid rows
    if "is_valid_row" in df.columns:
        report["invalid_row_count"] = int((~df["is_valid_row"]).sum())
        report["valid_row_count"] = int(df["is_valid_row"].sum())
        report["valid_row_percentage"] = (
            round(100 * report["valid_row_count"] / report["row_count"], 2)
            if report["row_count"] > 0
            else 0.0
        )

    # Count invalid IDs
    if "dq_invalid_ids" in df.columns:
        report["invalid_id_rows"] = int(df["dq_invalid_ids"].sum())

    # Identify columns with missing values
    null_pct = calculate_null_percentages(df)
    report["null_percentage_by_column"] = null_pct

    # Identify columns with any missing data
    missing_cols = [col for col, pct in null_pct.items() if pct > 0]
    report["missing_required_columns"] = missing_cols

    # Detect numeric outliers
    outliers = detect_numeric_outliers(df, numeric_columns)
    report["numeric_outlier_counts"] = outliers

    # Add DQ flag summary
    dq_flags = [
        "dq_missing_required_columns",
        "dq_invalid_ids",
        "dq_invalid_numeric",
        "dq_out_of_range",
    ]
    for flag in dq_flags:
        if flag in df.columns:
            report["data_quality_summary"][flag] = int(df[flag].sum())

    # Add additional info if provided
    if additional_info:
        report.update(additional_info)

    # Write to JSON file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation report written to {output_path}")
    logger.info(
        f"Report summary: {report['valid_row_count']}/{report['row_count']} valid rows"
    )

    return report


def generate_summary_statistics(
    df: pd.DataFrame, numeric_columns: List[str]
) -> Dict[str, Any]:
    """
    Generate summary statistics for numeric columns.

    Args:
        df: DataFrame to analyze
        numeric_columns: List of numeric column names

    Returns:
        Dictionary with summary statistics
    """
    stats = {}

    for col in numeric_columns:
        if col not in df.columns:
            continue

        data = df[col].dropna()
        if len(data) == 0:
            stats[col] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue

        stats[col] = {
            "count": int(data.count()),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "median": float(data.median()),
            "q25": float(data.quantile(0.25)),
            "q75": float(data.quantile(0.75)),
        }

    return stats


def generate_categorical_summary(
    df: pd.DataFrame, categorical_columns: List[str]
) -> Dict[str, Any]:
    """
    Generate summary statistics for categorical columns.

    Args:
        df: DataFrame to analyze
        categorical_columns: List of categorical column names

    Returns:
        Dictionary with categorical summaries
    """
    summary = {}

    for col in categorical_columns:
        if col not in df.columns:
            continue

        value_counts = df[col].value_counts()
        summary[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": value_counts.head(10).to_dict(),
            "null_count": int(df[col].isna().sum()),
        }

    return summary


def write_invalid_rows_report(
    df: pd.DataFrame, output_path: str, max_rows: int = 1000
) -> None:
    """
    Write invalid rows to CSV for inspection.

    Args:
        df: DataFrame with is_valid_row column
        output_path: Path to write CSV file
        max_rows: Maximum number of invalid rows to write
    """
    if "is_valid_row" not in df.columns:
        logger.warning("DataFrame does not have is_valid_row column")
        return

    invalid = df[~df["is_valid_row"]].copy()

    if len(invalid) == 0:
        logger.info("No invalid rows to write")
        return

    # Limit to max_rows
    if len(invalid) > max_rows:
        logger.warning(
            f"Limiting invalid rows report to {max_rows} rows (out of {len(invalid)})"
        )
        invalid = invalid.head(max_rows)

    # Write to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    invalid.to_csv(output_file, index=False)
    logger.info(f"Invalid rows written to {output_path} ({len(invalid)} rows)")


def print_validation_summary(report: Dict[str, Any]) -> None:
    """
    Print validation report summary to logger.

    Args:
        report: Validation report dictionary
    """
    logger.info("=" * 60)
    logger.info("VALIDATION REPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {report.get('timestamp', 'N/A')}")
    logger.info(f"Total rows: {report.get('row_count', 0)}")
    logger.info(
        f"Valid rows: {report.get('valid_row_count', 0)} ({report.get('valid_row_percentage', 0)}%)"
    )
    logger.info(f"Invalid rows: {report.get('invalid_row_count', 0)}")
    logger.info(f"Invalid ID rows: {report.get('invalid_id_rows', 0)}")

    if report.get("data_quality_summary"):
        logger.info("\nData Quality Issues:")
        for flag, count in report["data_quality_summary"].items():
            if count > 0:
                logger.info(f"  {flag}: {count}")

    if report.get("numeric_outlier_counts"):
        total_outliers = sum(report["numeric_outlier_counts"].values())
        if total_outliers > 0:
            logger.info(f"\nNumeric outliers detected: {total_outliers}")

    logger.info("=" * 60)
