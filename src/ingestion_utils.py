"""Utility functions supporting Phase 1 ingestion and normalization."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


LOGGER_NAME = "brightstar.phase1"


@dataclass
class WeekLookup:
    """Container that accelerates week label lookups."""

    mapping: pd.DataFrame

    def __post_init__(self) -> None:
        if self.mapping.empty:
            raise ValueError("Calendar map is empty. Phase 1 requires at least one mapping.")
        normalized = self.mapping.copy()
        normalized["raw_normalized"] = (
            normalized["raw_week_label"].astype(str).str.strip().str.lower()
        )
        self._lookup = normalized.set_index("raw_normalized")

    def resolve(self, raw_label: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Return normalized label and boundaries for a raw week label."""

        key = str(raw_label).strip().lower()
        if not key:
            return "", None, None
        if key in self._lookup.index:
            row = self._lookup.loc[key]
            return (
                str(row["week_label"]),
                _safe_date(row.get("week_start")),
                _safe_date(row.get("week_end")),
            )
        return str(raw_label), None, None


def _safe_date(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_config(path: str | Path) -> Dict:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def ensure_directory(directory: str | Path) -> Path:
    """Ensure that the directory exists."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(config: Dict) -> logging.Logger:
    """Configure project-wide logging for the ingestion phase."""

    log_dir = ensure_directory(config["paths"]["logs_dir"])
    file_name = config.get("logging", {}).get("file_name", "phase1_ingestion.log")
    log_path = log_dir / file_name

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, config.get("logging", {}).get("level", "INFO")))
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.debug("Logging initialised. Logs will be written to %s", log_path)
    return logger


def load_calendar_map(path: str | Path) -> WeekLookup:
    """Load the calendar reference table."""

    df = pd.read_csv(path)
    required_columns = {"raw_week_label", "week_label", "week_start", "week_end"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Calendar map missing required columns: {missing}")
    return WeekLookup(df)


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize header casing and whitespace."""

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def detect_structure(df: pd.DataFrame, id_columns: Iterable[str]) -> str:
    """Detect whether the dataset is wide or long format."""

    columns = set(df.columns)
    if {"Week", "Metric", "Value"}.issubset(columns):
        return "long"
    if {"week", "metric", "value"}.issubset({col.lower() for col in df.columns}):
        return "long"
    if all(col in df.columns for col in id_columns):
        non_id = [col for col in df.columns if col not in id_columns]
        for column in non_id:
            _, week = split_metric_and_week(str(column))
            if week:
                return "wide"
        if len(non_id) > 3:
            return "wide"
    return "long"


def split_metric_and_week(column_name: str) -> Tuple[str, Optional[str]]:
    """Split a column name into metric and raw week parts."""

    clean_name = str(column_name).strip()
    paren_matches = re.findall(r"\(([^()]*)\)", clean_name)
    if paren_matches:
        week_token = paren_matches[-1].strip()
        metric_part = clean_name[: clean_name.rfind(f"({paren_matches[-1]})")].strip()
        if not metric_part:
            metric_part = clean_name
        if week_token:
            return metric_part.strip(), week_token

    range_match = re.search(
        r"(\d{2}/\d{2}/\d{4})\s*[-–]\s*(\d{2}/\d{2}/\d{4})",
        clean_name,
    )
    if range_match:
        return clean_name[: range_match.start()].strip(), f"{range_match.group(1)}-{range_match.group(2)}"

    week_number_match = re.search(r"(\b(?:FW|W)\s?\d{1,2}\s?\d{4}\b|\d{4}-W\d{2})", clean_name)
    if week_number_match:
        token = week_number_match.group(1)
        metric_part = clean_name.replace(token, "").strip(" -_")
        return metric_part.strip(), token.strip()

    return clean_name, None


def melt_wide_dataframe(
    df: pd.DataFrame,
    id_columns: List[str],
    lookup: WeekLookup,
    value_round: Optional[int] = None,
) -> pd.DataFrame:
    """Transform a wide dataframe into the normalized long format."""

    records: List[Dict[str, object]] = []
    for column in df.columns:
        if column in id_columns:
            continue
        metric, raw_week = split_metric_and_week(column)
        if raw_week is None:
            continue
        week_label, week_start, week_end = lookup.resolve(raw_week)
        for _, row in df.iterrows():
            value = row[column]
            if pd.isna(value):
                continue
            if value_round is not None and isinstance(value, (int, float)):
                value = round(float(value), value_round)
            records.append(
                {
                    "vendor_code": row[id_columns[0]],
                    "asin": row[id_columns[1]],
                    "metric": metric,
                    "week_label": week_label,
                    "week_raw": raw_week,
                    "week_start": week_start,
                    "week_end": week_end,
                    "value": value,
                }
            )
    return pd.DataFrame.from_records(records)


def normalize_long_dataframe(
    df: pd.DataFrame,
    lookup: WeekLookup,
    week_column: str,
    metric_column: str,
    value_column: str,
    id_columns: List[str],
    value_round: Optional[int] = None,
) -> pd.DataFrame:
    """Normalize a long dataframe into the canonical schema."""

    df = df.copy()
    df.rename(columns={week_column: "week_raw"}, inplace=True)
    df.rename(columns={metric_column: "metric"}, inplace=True)
    df.rename(columns={value_column: "value"}, inplace=True)
    df.rename(columns={id_columns[0]: "vendor_code", id_columns[1]: "asin"}, inplace=True)

    def map_week(raw_week: object) -> Tuple[str, Optional[str], Optional[str]]:
        return lookup.resolve(str(raw_week))

    weeks = df["week_raw"].apply(map_week)
    df["week_label"] = weeks.apply(lambda x: x[0])
    df["week_start"] = weeks.apply(lambda x: x[1])
    df["week_end"] = weeks.apply(lambda x: x[2])

    if value_round is not None:
        df["value"] = pd.to_numeric(df["value"], errors="coerce").round(value_round)
    else:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df[["vendor_code", "asin", "metric", "week_label", "week_raw", "week_start", "week_end", "value"]]


def deduplicate_records(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Drop duplicate records keeping the most recent entry."""

    before = len(df)
    deduped = df.drop_duplicates(
        subset=["vendor_code", "asin", "metric", "week_label"], keep="last"
    )
    after = len(deduped)
    if after < before:
        logger.info("Deduplicated %s duplicate rows", before - after)
    return deduped


def detect_week_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to find a column representing the week label."""

    candidates = [
        "week",
        "Week",
        "Week Label",
        "Fiscal Week",
        "week_label",
        "Week_Number",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def detect_metric_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to find a metric column for long-form data."""

    candidates = ["Metric", "metric", "Measure", "measure"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def detect_value_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to find a value column for long-form data."""

    candidates = ["Value", "value", "Amount", "amount"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def write_output(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> Path:
    """Persist the normalized dataframe to disk."""

    output_path = Path(config["paths"]["normalized_output"])
    ensure_directory(output_path.parent)
    fmt = config.get("ingestion", {}).get("output_format", "parquet").lower()

    try:
        if fmt == "csv":
            df.to_csv(output_path.with_suffix(".csv"), index=False)
            return output_path.with_suffix(".csv")
        if fmt == "parquet":
            df.to_parquet(output_path, index=False)
            return output_path
    except Exception as exc:  # pragma: no cover - fallback path is environment dependent
        logger.warning("Failed to write %s due to %s. Falling back to CSV.", fmt, exc)
        fallback = output_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback

    fallback = output_path.with_suffix(".csv")
    df.to_csv(fallback, index=False)
    return fallback


def write_state(metadata: Dict[str, object], config: Dict) -> Path:
    """Persist ingestion state metadata as JSON."""

    state_path = Path(config["paths"]["ingestion_state"])
    ensure_directory(state_path.parent)
    with open(state_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return state_path


def summarize_phase(df: pd.DataFrame, template: str) -> str:
    """Generate a human-readable summary of the ingestion results."""

    if df.empty:
        return "No rows processed."
    summary = template.format(
        total_rows=len(df),
        unique_weeks=df["week_label"].nunique(),
        last_week=df.sort_values("week_start", na_position="last")["week_label"].iloc[-1],
    )
    return summary


def validate_extension(path: Path, allowed: Iterable[str]) -> None:
    """Ensure the file extension is allowed."""

    suffix = path.suffix.lower()
    if suffix not in {ext.lower() for ext in allowed}:
        raise ValueError(f"Unsupported file extension: {suffix}. Allowed: {allowed}")


def read_input_file(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)
