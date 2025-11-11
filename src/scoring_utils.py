"""Utility helpers for Phase 2 scoring computations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .ingestion_utils import ensure_directory


LOGGER_NAME = "brightstar.phase2"


@dataclass
class MetricConfig:
    """Configuration describing how to score a single metric."""

    name: str
    weight: float
    direction: str

    def is_descending(self) -> bool:
        """Return ``True`` when lower values indicate better performance."""

        return self.direction.lower() in {"down", "desc", "lower", "decrease"}


def setup_logging(config: Dict) -> logging.Logger:
    """Configure logging for the scoring phase."""

    log_dir = ensure_directory(config["paths"]["logs_dir"])
    file_name = config.get("logging", {}).get("scoring_file_name", "phase2_scoring.log")
    log_path = log_dir / file_name

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, config.get("logging", {}).get("scoring_level", "INFO")))
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.debug("Phase 2 logging initialised at %s", log_path)
    return logger


def load_normalized_input(config: Dict, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load the normalized dataset produced by Phase 1."""

    scoring_config = config.get("scoring", {})
    candidate_paths: List[Path] = []

    normalized_preference = scoring_config.get("normalized_input")
    if normalized_preference:
        candidate_paths.append(Path(normalized_preference))

    default_normalized = config["paths"].get("normalized_output")
    if default_normalized:
        candidate_paths.append(Path(default_normalized))

    # fall back to CSV representation when parquet is not available
    candidate_paths.append(Path(config["paths"]["processed_dir"]) / "normalized_phase1.csv")

    for path in candidate_paths:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                try:
                    return pd.read_parquet(path)
                except Exception as exc:  # pragma: no cover - pandas engine coverage is environment dependent
                    if logger:
                        logger.warning("Unable to read parquet file %s (%s). Falling back to CSV if available.", path, exc)
                    continue
            if path.suffix.lower() in {".csv", ".txt"}:
                return pd.read_csv(path)

    raise FileNotFoundError(
        "Unable to locate normalized dataset. Ensure Phase 1 output exists before running Phase 2."
    )


def build_metric_map(config: Dict) -> Dict[str, MetricConfig]:
    """Construct the metric configuration mapping from settings."""

    scoring_config = config.get("scoring", {})
    weights = scoring_config.get("metric_weights", {})
    directions = scoring_config.get("metric_directions", {})
    if not weights:
        raise ValueError("No metric weights configured for scoring.")

    metric_map = {}
    for name, weight in weights.items():
        direction = directions.get(name, "up")
        metric_map[name] = MetricConfig(name=name, weight=float(weight), direction=str(direction))
    return metric_map


def aggregate_metrics(
    df: pd.DataFrame,
    entity_columns: Iterable[str],
    metric_column: str = "metric",
    value_column: str = "value",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Aggregate metric values per entity according to the configured function."""

    if df.empty:
        return pd.DataFrame(columns=[*entity_columns, metric_column, "Raw_Value"])

    group_cols = list(entity_columns) + [metric_column]
    grouped = df.groupby(group_cols)[value_column]

    if aggregation == "mean":
        values = grouped.mean()
    elif aggregation == "median":
        values = grouped.median()
    elif aggregation == "sum":
        values = grouped.sum()
    else:
        raise ValueError(f"Unsupported aggregation function: {aggregation}")

    aggregated = values.reset_index().rename(columns={value_column: "Raw_Value"})
    return aggregated


def ensure_metric_coverage(
    aggregated: pd.DataFrame,
    entity_columns: Iterable[str],
    metric_map: Dict[str, MetricConfig],
) -> pd.DataFrame:
    """Ensure every entity has an entry for each configured metric."""

    entity_cols = list(entity_columns)
    metrics = list(metric_map.keys())
    if aggregated.empty:
        columns = entity_cols + ["metric", "Raw_Value"]
        return pd.DataFrame(columns=columns)

    unique_entities = aggregated[entity_cols].drop_duplicates()
    entity_cartesian = unique_entities.assign(_key=1).merge(
        pd.DataFrame({"metric": metrics, "_key": 1}), on="_key"
    ).drop(columns="_key")

    merged = entity_cartesian.merge(aggregated, on=entity_cols + ["metric"], how="left")
    merged["Raw_Value"] = merged["Raw_Value"].astype(float)
    return merged


def compute_normalised_scores(
    aggregated: pd.DataFrame,
    metric_map: Dict[str, MetricConfig],
    fallback_normalised: float,
) -> pd.DataFrame:
    """Add normalised and weighted scores for each metric."""

    if aggregated.empty:
        aggregated = aggregated.copy()
        aggregated["Normalized_Value"] = pd.Series(dtype=float)
        aggregated["Weight"] = pd.Series(dtype=float)
        aggregated["Weighted_Score"] = pd.Series(dtype=float)
        return aggregated

    metric_stats = (
        aggregated.groupby("metric")["Raw_Value"].agg(["min", "max"]).rename(columns={"min": "_min", "max": "_max"})
    )

    def normalise(row: pd.Series) -> float:
        config = metric_map.get(row["metric"])
        if config is None:
            return fallback_normalised

        raw_value = row["Raw_Value"]
        if pd.isna(raw_value):
            base = fallback_normalised
        else:
            stats = metric_stats.loc[row["metric"]]
            min_val = stats["_min"]
            max_val = stats["_max"]
            if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
                base = 0.5
            else:
                base = (raw_value - min_val) / (max_val - min_val)
                base = max(0.0, min(1.0, base))

        if config.is_descending():
            base = 1.0 - base

        return max(0.0, min(1.0, base))

    aggregated = aggregated.copy()
    aggregated["Normalized_Value"] = aggregated.apply(normalise, axis=1).round(4)
    aggregated["Weight"] = aggregated["metric"].map(lambda m: metric_map.get(m, MetricConfig(m, 0.0, "up")).weight)
    aggregated["Weighted_Score"] = (aggregated["Normalized_Value"] * aggregated["Weight"]).round(4)
    return aggregated


def compute_composite_scores(
    scored: pd.DataFrame,
    metric_map: Dict[str, MetricConfig],
    composite_scale: float,
    entity_columns: Iterable[str],
) -> pd.DataFrame:
    """Aggregate weighted scores into composite entity scores."""

    if scored.empty:
        return scored.assign(Composite_Score=pd.Series(dtype=float))

    total_weight = sum(config.weight for config in metric_map.values())
    if total_weight <= 0:
        raise ValueError("Metric weights must sum to a positive value.")

    entity_cols = list(entity_columns)

    composite = (
        scored.groupby(entity_cols)["Weighted_Score"].sum().reset_index().rename(columns={"Weighted_Score": "_weighted_sum"})
    )
    composite["Composite_Score"] = ((composite["_weighted_sum"] / total_weight) * composite_scale).round(3)
    scored = scored.merge(composite.drop(columns=["_weighted_sum"]), on=entity_cols, how="left")
    return scored


def assign_performance_category(scored: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Assign qualitative categories based on composite score thresholds."""

    if scored.empty or "Composite_Score" not in scored.columns:
        scored["Performance_Category"] = pd.Series(dtype=str)
        return scored

    ordered_thresholds = sorted(((label, value) for label, value in thresholds.items()), key=lambda item: item[1], reverse=True)

    def label_score(score: float) -> str:
        for label, value in ordered_thresholds:
            if score >= value:
                return label.title()
        return ordered_thresholds[-1][0].title() if ordered_thresholds else "Unclassified"

    scored = scored.copy()
    scored["Performance_Category"] = scored["Composite_Score"].apply(label_score)
    return scored


def evaluate_improvement_flags(
    scored: pd.DataFrame,
    improvement_rules: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Apply rule-based improvement flags per metric."""

    if scored.empty:
        scored["Improvement_Flag"] = pd.Series(dtype=str)
        return scored

    default_rule = improvement_rules.get("default", {})

    def apply_rule(row: pd.Series) -> Optional[str]:
        rule = improvement_rules.get(row["metric"], default_rule)
        if not rule:
            return None

        normalized_value = row.get("Normalized_Value")
        raw_value = row.get("Raw_Value")

        if "normalized_lt" in rule and normalized_value < rule["normalized_lt"]:
            return rule.get("label", "Watch")
        if "normalized_gt" in rule and normalized_value > rule["normalized_gt"]:
            return rule.get("label", "Watch")
        if "raw_lt" in rule and raw_value < rule["raw_lt"]:
            return rule.get("label", "Watch")
        if "raw_gt" in rule and raw_value > rule["raw_gt"]:
            return rule.get("label", "Watch")
        return None

    scored = scored.copy()
    scored["Improvement_Flag"] = scored.apply(apply_rule, axis=1)
    return scored


def add_rankings(scored: pd.DataFrame, entity_columns: Iterable[str]) -> pd.DataFrame:
    """Calculate rankings per entity type, metric, and category."""

    if scored.empty:
        scored["Rank_Overall"] = pd.Series(dtype=float)
        scored["Rank_By_Metric"] = pd.Series(dtype=float)
        scored["Rank_By_Category"] = pd.Series(dtype=float)
        return scored

    scored = scored.copy()
    entity_cols = list(entity_columns)
    entity_type_col = entity_cols[0] if entity_cols else "entity_type"

    unique_overall = scored.drop_duplicates(subset=entity_cols + ["Composite_Score"])[entity_cols + ["Composite_Score"]]
    unique_overall["Rank_Overall"] = unique_overall.groupby(entity_type_col)["Composite_Score"].rank(
        ascending=False, method="dense"
    )
    scored = scored.merge(unique_overall[entity_cols + ["Rank_Overall"]], on=entity_cols, how="left")

    metric_unique = scored.drop_duplicates(subset=entity_cols + ["metric"])[entity_cols + ["metric", "Normalized_Value"]]
    metric_unique["Rank_By_Metric"] = metric_unique.groupby([entity_type_col, "metric"])["Normalized_Value"].rank(
        ascending=False, method="dense"
    )
    scored = scored.merge(
        metric_unique[entity_cols + ["metric", "Rank_By_Metric"]],
        on=entity_cols + ["metric"],
        how="left",
    )

    if "Performance_Category" in scored.columns:
        category_unique = scored.drop_duplicates(subset=entity_cols + ["Performance_Category"])[
            entity_cols + ["Performance_Category", "Composite_Score"]
        ]
        category_unique["Rank_By_Category"] = category_unique.groupby([entity_type_col, "Performance_Category"])[
            "Composite_Score"
        ].rank(ascending=False, method="dense")
        scored = scored.merge(
            category_unique[entity_cols + ["Performance_Category", "Rank_By_Category"]],
            on=entity_cols + ["Performance_Category"],
            how="left",
        )
    else:
        scored["Rank_By_Category"] = pd.NA

    return scored


def persist_outputs(
    scored: pd.DataFrame,
    config: Dict,
    entity_type: str,
    logger: logging.Logger,
) -> Tuple[Path, Optional[Path]]:
    """Write scorecard results for the specified entity type."""

    ensure_directory(config["paths"]["scorecards_dir"])

    if entity_type == "vendor":
        csv_path = Path(config["paths"]["scorecard_vendor_csv"])
        parquet_path = Path(config["paths"]["scorecard_vendor_parquet"])
    else:
        csv_path = Path(config["paths"]["scorecard_asin_csv"])
        parquet_path = Path(config["paths"]["scorecard_asin_parquet"])

    scored.to_csv(csv_path, index=False)
    try:
        scored.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Unable to write parquet scorecard %s (%s).", parquet_path, exc)
        parquet_path = None

    return csv_path, parquet_path


def write_audit(audit: pd.DataFrame, config: Dict) -> Path:
    """Persist the consolidated score audit dataframe."""

    audit_path = Path(config["paths"]["scorecard_audit_csv"])
    ensure_directory(audit_path.parent)
    audit.to_csv(audit_path, index=False)
    return audit_path


def write_metadata(metadata: Dict, config: Dict) -> Path:
    """Persist run metadata for auditability."""

    metadata_path = Path(config["paths"]["scorecard_metadata"])
    ensure_directory(metadata_path.parent)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def summarize_entities(scored: pd.DataFrame, entity_column: str, top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top and bottom performers for console summaries."""

    if scored.empty:
        empty = pd.DataFrame(columns=[entity_column, "Composite_Score"])
        return empty, empty

    summary_cols = [entity_column, "Composite_Score", "Performance_Category"]
    sorted_df = (
        scored.drop_duplicates(subset=summary_cols)
        .loc[:, summary_cols]
        .sort_values("Composite_Score", ascending=False)
    )
    top = sorted_df.head(top_n)
    bottom = sorted_df.tail(top_n).iloc[::-1]
    return top, bottom
