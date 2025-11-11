"""Phase 1 ingestion and normalization entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .ingestion_utils import (
    deduplicate_records,
    detect_metric_column,
    detect_structure,
    detect_value_column,
    detect_week_column,
    ensure_directory,
    load_calendar_map,
    load_config,
    melt_wide_dataframe,
    normalize_headers,
    normalize_long_dataframe,
    read_input_file,
    setup_logging,
    summarize_phase,
    validate_extension,
    write_output,
    write_state,
)


def resolve_id_columns(df: pd.DataFrame, config: dict) -> List[str]:
    """Determine identifier columns using config indices or explicit names."""

    id_columns = config["ingestion"].get("id_columns", [])
    if all(col in df.columns for col in id_columns):
        return id_columns

    vendor_idx = config["ingestion"].get("vendor_column", 0)
    asin_idx = config["ingestion"].get("asin_column", 1)
    columns = list(df.columns)
    try:
        return [columns[vendor_idx], columns[asin_idx]]
    except IndexError as exc:  # pragma: no cover - configuration error should surface early
        raise ValueError("Unable to resolve identifier columns from configuration.") from exc


def ingest_file(path: Path, config: dict, lookup) -> pd.DataFrame:
    """Normalize a single raw export file."""

    allowed = config["ingestion"].get("allowed_extensions", [".csv", ".xlsx"])
    validate_extension(path, allowed)
    df_raw = read_input_file(path)
    df_raw = normalize_headers(df_raw)
    id_columns = resolve_id_columns(df_raw, config)

    structure = detect_structure(df_raw, id_columns)
    value_round = config["ingestion"].get("value_round")

    if structure == "wide" and not config["ingestion"].get("force_long_format", False):
        normalized = melt_wide_dataframe(
            df_raw,
            id_columns=id_columns,
            lookup=lookup,
            value_round=value_round,
        )
    else:
        week_column = detect_week_column(df_raw)
        metric_column = detect_metric_column(df_raw)
        value_column = detect_value_column(df_raw)
        if not week_column or not metric_column or not value_column:
            raise ValueError(
                f"Unable to detect columns for long format in {path}. Found week={week_column}, metric={metric_column}, value={value_column}"
            )
        normalized = normalize_long_dataframe(
            df_raw,
            lookup=lookup,
            week_column=week_column,
            metric_column=metric_column,
            value_column=value_column,
            id_columns=id_columns,
            value_round=value_round,
        )

    normalized["source_file"] = path.name
    return normalized


def gather_input_files(config: dict, explicit: Optional[Iterable[Path]] = None) -> List[Path]:
    """Collect all files that should be ingested."""

    if explicit:
        return [Path(p) for p in explicit]
    raw_dir = Path(config["paths"]["raw_dir"])
    ensure_directory(raw_dir)
    allowed = {ext.lower() for ext in config["ingestion"].get("allowed_extensions", [])}
    files = [
        path
        for path in raw_dir.iterdir()
        if path.suffix.lower() in allowed
    ]
    return sorted(files)


def run_phase1(config_path: str = "config.yaml", input_files: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Execute the ingestion pipeline and return the normalized dataframe."""

    config = load_config(config_path)
    logger = setup_logging(config)
    lookup = load_calendar_map(config["paths"]["calendar_map"])

    file_paths = gather_input_files(config, [Path(p) for p in input_files] if input_files else None)
    if not file_paths:
        logger.warning("No input files found for Phase 1 ingestion.")
        return pd.DataFrame(columns=[
            "vendor_code",
            "asin",
            "metric",
            "week_label",
            "week_raw",
            "week_start",
            "week_end",
            "value",
            "source_file",
        ])

    frames = []
    for path in file_paths:
        logger.info("Ingesting %s", path)
        normalized = ingest_file(path, config, lookup)
        frames.append(normalized)

    combined = pd.concat(frames, ignore_index=True)
    combined = deduplicate_records(combined, logger)
    combined.sort_values(["vendor_code", "asin", "metric", "week_start", "week_label"], inplace=True)

    output_path = write_output(combined, config, logger)
    logger.info("Normalized dataset written to %s", output_path)

    metadata = {
        "records": int(len(combined)),
        "vendors": combined["vendor_code"].nunique(),
        "asins": combined["asin"].nunique(),
        "weeks": combined["week_label"].nunique(),
        "last_week": combined.sort_values("week_start", na_position="last")["week_label"].iloc[-1]
        if not combined.empty
        else None,
        "output_path": str(output_path),
    }
    write_state(metadata, config)

    template = config.get("summary", {}).get(
        "template", "Processed {total_rows} rows across {unique_weeks} weeks. Last week: {last_week}."
    )
    summary = summarize_phase(combined, template)
    if config.get("summary", {}).get("enabled", True):
        print(summary)
    logger.info(summary)
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    """Create a CLI argument parser for the phase runner."""

    parser = argparse.ArgumentParser(description="Brightstar Phase 1 ingestion")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file")
    parser.add_argument("inputs", nargs="*", help="Explicit input files to ingest")
    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args()
    run_phase1(config_path=args.config, input_files=args.inputs or None)


if __name__ == "__main__":
    main()
