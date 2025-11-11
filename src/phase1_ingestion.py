"""Phase 1 ingestion and normalization entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .ingestion_utils import (
    apply_master_lookup,
    deduplicate_records,
    detect_metric_column,
    detect_structure,
    detect_value_column,
    detect_week_column,
    detect_identifier_column,
    ensure_directory,
    load_calendar_map,
    load_master_lookup,
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


def resolve_id_columns(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, List[str]]:
    """Determine identifier columns using configured aliases or positional hints."""

    ingestion_config = config.get("ingestion", {})
    explicit = ingestion_config.get("id_columns", [])
    if explicit and all(col in df.columns for col in explicit):
        return df, explicit

    aliases = ingestion_config.get("column_aliases", {})
    asin_aliases: Iterable[str] = aliases.get("asin", ["asin"])
    vendor_aliases: Iterable[str] = aliases.get("vendor", ["vendor", "vendor_code"])

    vendor_fallback = ingestion_config.get("vendor_column")
    asin_fallback = ingestion_config.get("asin_column")

    working_df, vendor_column = detect_identifier_column(
        df,
        vendor_aliases,
        fallback_index=vendor_fallback,
        default_name="Vendor",
    )

    working_df, asin_column = detect_identifier_column(
        working_df,
        asin_aliases,
        fallback_index=asin_fallback,
        default_name="ASIN",
    )

    if asin_column is None:
        raise ValueError("Unable to detect ASIN column. Provide aliases in configuration or ensure header includes ASIN.")

    if vendor_column is None:
        working_df = working_df.copy()
        vendor_column = "Vendor"
        working_df[vendor_column] = pd.NA

    return working_df, [vendor_column, asin_column]


def ingest_file(path: Path, config: dict, lookup, master_lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize a single raw export file."""

    allowed = config["ingestion"].get("allowed_extensions", [".csv", ".xlsx"])
    validate_extension(path, allowed)
    df_raw = read_input_file(path)
    df_raw = normalize_headers(df_raw)
    df_raw, id_columns = resolve_id_columns(df_raw, config)

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
    if master_lookup is not None:
        normalized = apply_master_lookup(normalized, master_lookup)
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
    master_lookup = load_master_lookup(
        config["paths"].get("masterfile"),
        config.get("masterfile"),
    )

    file_paths = gather_input_files(config, [Path(p) for p in input_files] if input_files else None)
    if not file_paths:
        logger.warning("No input files found for Phase 1 ingestion.")
        return pd.DataFrame(columns=[
            "vendor_code",
            "vendor_name",
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
        normalized = ingest_file(path, config, lookup, master_lookup)
        frames.append(normalized)

    combined = pd.concat(frames, ignore_index=True)
    combined = deduplicate_records(combined, logger)
    combined.sort_values(
        ["vendor_code", "vendor_name", "asin", "metric", "week_start", "week_label"],
        inplace=True,
    )

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
