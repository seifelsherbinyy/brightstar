"""Utility functions supporting Phase 1 ingestion and normalization."""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml



LOGGER_NAME = "brightstar.phase1"


DATE_RANGE_PATTERN = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{2,4})\s*[-–]\s*(\d{1,2}/\d{1,2}/\d{2,4})"
)
WEEK_NUMBER_PATTERN = re.compile(
    r"(?:FW|W)\s*0?(\d{1,2})(?:[^\d]*(\d{4}))?",
    flags=re.IGNORECASE,
)
ISO_WEEK_PATTERN = re.compile(r"(\d{4})[-_/]?W?0?(\d{1,2})")


@dataclass
class WeekLookup:
    """Container that accelerates week label lookups."""

    mapping: pd.DataFrame
    _cache: Dict[str, Tuple[str, Optional[str], Optional[str]]] = field(default_factory=dict)

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

        key = str(raw_label).strip()
        normalized_key = key.lower()
        if not normalized_key:
            return "", None, None

        if normalized_key in self._cache:
            return self._cache[normalized_key]

        if normalized_key in self._lookup.index:
            row = self._lookup.loc[normalized_key]
            result = (
                str(row["week_label"]),
                _safe_date(row.get("week_start")),
                _safe_date(row.get("week_end")),
            )
            self._cache[normalized_key] = result
            return result

        parsed = _parse_week_from_label(key)
        if parsed:
            self._cache[normalized_key] = parsed
            return parsed

        result = (key, None, None)
        self._cache[normalized_key] = result
        return result


def _safe_date(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_week_from_label(raw_label: str) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """Parse date ranges or explicit week references from a raw label."""

    text = str(raw_label).strip()
    if not text:
        return None

    range_match = DATE_RANGE_PATTERN.search(text)
    if range_match:
        start_raw, end_raw = range_match.groups()
        start_dt = pd.to_datetime(start_raw, errors="coerce")
        end_dt = pd.to_datetime(end_raw, errors="coerce")
        if pd.notna(start_dt) and pd.notna(end_dt):
            iso = start_dt.isocalendar()
            week_label, iso_start, iso_end = _week_from_number(iso.year, iso.week)
            start_value = iso_start or start_dt.strftime("%Y-%m-%d")
            end_value = iso_end or end_dt.strftime("%Y-%m-%d")
            return week_label, start_value, end_value

    iso_match = ISO_WEEK_PATTERN.search(text)
    if iso_match:
        year = int(iso_match.group(1))
        week = int(iso_match.group(2))
        return _week_from_number(year, week)

    week_match = WEEK_NUMBER_PATTERN.search(text)
    if week_match:
        week = int(week_match.group(1))
        year = week_match.group(2)
        if year:
            return _week_from_number(int(year), week)

    return None


def _week_from_number(year: int, week: int) -> Tuple[str, str, str]:
    label = f"{year}W{week:02d}"
    try:
        start = date.fromisocalendar(year, week, 1)
        end = date.fromisocalendar(year, week, 7)
    except ValueError:
        return label, None, None
    return label, start.isoformat(), end.isoformat()


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


def load_master_lookup(path: str | Path, config: Dict | None = None) -> pd.DataFrame:
    """Load the master reference file linking ASINs to vendor details."""

    if not path:
        return pd.DataFrame(columns=["asin", "vendor_code", "vendor_name"])

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["asin", "vendor_code", "vendor_name"])

    config = config or {}
    sheet_name = config.get("sheet_name")
    columns = config.get("columns", {})

    # Support both .xlsx and .csv with multi-sheet handling
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        result = pd.read_excel(path, sheet_name=sheet_name)
        
        # If result is a dict of sheets, pick the right one
        if isinstance(result, dict):
            if sheet_name and sheet_name in result:
                frame = result[sheet_name]
            elif "Info" in result:
                frame = result["Info"]
            elif "Info2" in result:
                frame = result["Info2"]
            else:
                # Pick the first sheet
                frame = next(iter(result.values()))
        else:
            frame = result
    else:
        frame = pd.read_csv(path)
    
    if columns:
        rename_map = {value: key for key, value in columns.items()}
        frame = frame.rename(columns=rename_map)

    expected = {"asin", "vendor_code", "vendor_name"}
    # Case-insensitive column rename (handle spaces and underscores)
    lower_map = {c.lower().replace(' ', '_'): c for c in frame.columns}
    rename_map = {}
    for key in expected:
        if key not in frame.columns and key in lower_map:
            rename_map[lower_map[key]] = key
    if rename_map:
        frame = frame.rename(columns=rename_map)

    frame = frame[[c for c in frame.columns if c in expected]].copy()
    for c in ("asin", "vendor_code", "vendor_name"):
        if c not in frame.columns:
            frame[c] = pd.NA
        else:
            frame[c] = frame[c].astype('string').str.strip()

    frame = frame.dropna(subset=['asin']).drop_duplicates(subset=['asin'], keep='last')
    return frame[["asin", "vendor_code", "vendor_name"]]


def apply_master_lookup(df: pd.DataFrame, master_lookup: pd.DataFrame) -> pd.DataFrame:
    """Enrich normalized rows with vendor metadata from the master lookup."""

    if master_lookup.empty:
        if "vendor_name" not in df.columns:
            df = df.copy()
            df["vendor_name"] = pd.NA
        return df

    normalized = df.copy()
    normalized["asin"] = normalized["asin"].astype(str).str.strip()
    enriched = normalized.merge(master_lookup, on="asin", how="left", suffixes=("", "_master"))

    if "vendor_code_master" in enriched.columns:
        enriched["vendor_code"] = enriched["vendor_code_master"].where(
            enriched["vendor_code_master"].notna() & (enriched["vendor_code_master"].astype(str).str.len() > 0),
            enriched["vendor_code"],
        )
        enriched.drop(columns=["vendor_code_master"], inplace=True)

    if "vendor_name_master" in enriched.columns:
        enriched["vendor_name"] = enriched["vendor_name_master"].where(
            enriched["vendor_name_master"].notna() & (enriched["vendor_name_master"].astype(str).str.len() > 0),
            enriched.get("vendor_name"),
        )
        enriched.drop(columns=["vendor_name_master"], inplace=True)

    if "vendor_name" not in enriched.columns:
        enriched["vendor_name"] = pd.NA
    else:
        enriched["vendor_name"] = enriched["vendor_name"].apply(
            lambda value: (str(value).strip() or pd.NA)
            if pd.notna(value)
            else pd.NA
        )

    enriched["vendor_code"] = enriched["vendor_code"].apply(
        lambda value: (str(value).strip() or pd.NA) if pd.notna(value) else pd.NA
    )

    return enriched


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize header casing and whitespace."""

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _normalize_header_token(token: str) -> str:
    """Normalize a single header token for case-insensitive matching."""
    return str(token).strip().lower().replace(" ", "_").replace("-", "_")


def detect_identifier_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    fallback_index: Optional[int] = None,
    default_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Detect a column by alias and optionally create it when missing."""

    # Build normalized header map once
    normalized_map = {_normalize_header_token(col): col for col in df.columns}

    # For each candidate alias, normalize the alias token and look it up
    for alias in aliases:
        key = _normalize_header_token(alias)
        if key in normalized_map:
            detected = normalized_map[key]
            series = df[detected].astype('string').str.strip()
            if series.notna().any():
                return df, detected

    if fallback_index is not None:
        columns = list(df.columns)
        if -len(columns) <= fallback_index < len(columns):
            return df, columns[fallback_index]

    if default_name and default_name not in df.columns:
        df = df.copy()
        df[default_name] = pd.NA
        return df, default_name

    return df, default_name if default_name in df.columns else None


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
        for token in reversed(paren_matches):
            candidate = token.strip()
            if not candidate:
                continue
            if not re.search(r"[A-Za-z0-9]", candidate):
                continue
            metric_part = clean_name[: clean_name.rfind(f"({token})")].strip()
            if not metric_part:
                metric_part = clean_name
            if _parse_week_from_label(candidate) or re.search(r"\bW\d{1,2}\b", candidate, re.IGNORECASE) or re.search(r"\d", candidate):
                return metric_part.strip(), candidate

    range_match = DATE_RANGE_PATTERN.search(clean_name)
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
            csv_path = output_path.with_suffix(".csv")
            df.to_csv(
                csv_path,
                index=False,
                encoding='utf-8',
                lineterminator='\n',
                quoting=csv.QUOTE_MINIMAL
            )
            return csv_path
        if fmt == "parquet":
            df.to_parquet(output_path, index=False)
            return output_path
    except Exception as exc:  # pragma: no cover - fallback path is environment dependent
        logger.warning("Failed to write %s due to %s. Falling back to CSV.", fmt, exc)
        fallback = output_path.with_suffix(".csv")
        df.to_csv(
            fallback,
            index=False,
            encoding='utf-8',
            lineterminator='\n',
            quoting=csv.QUOTE_MINIMAL
        )
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
