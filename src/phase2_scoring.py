"""Phase 2 scoring engine entry point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .ingestion_utils import load_config
from .scoring_utils import (
    add_rankings,
    aggregate_metrics,
    assign_performance_category,
    build_metric_map,
    compute_composite_scores,
    compute_normalised_scores,
    ensure_metric_coverage,
    evaluate_improvement_flags,
    load_normalized_input,
    persist_outputs,
    setup_logging,
    summarize_entities,
    write_audit,
    write_metadata,
)
def _prepare_entity_frame(
    normalized_df: pd.DataFrame,
    metric_map: Dict[str, object],
    entity_type: str,
    aggregation: str,
    fallback_normalised: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate and score metrics for the requested entity level."""

    frame = normalized_df.copy()
    frame["entity_type"] = entity_type

    entity_columns: list[str] = ["entity_type", "vendor_code"]
    if "vendor_name" in frame.columns and frame["vendor_name"].notna().any():
        entity_columns.append("vendor_name")

    if entity_type != "vendor":
        entity_columns.append("asin")

    aggregated = aggregate_metrics(frame, entity_columns, aggregation=aggregation)
    aggregated = aggregated[aggregated["metric"].isin(metric_map.keys())]
    aggregated = ensure_metric_coverage(aggregated, entity_columns, metric_map)
    aggregated = compute_normalised_scores(
        aggregated,
        metric_map=metric_map,
        fallback_normalised=fallback_normalised,
    )
    return aggregated, entity_columns


def _apply_scoring(
    scored: pd.DataFrame,
    metric_map: Dict[str, object],
    composite_scale: float,
    thresholds: Dict[str, float],
    improvement_rules: Dict[str, Dict[str, float]],
    entity_columns: Iterable[str],
) -> pd.DataFrame:
    """Compute composite scores, rankings, and qualitative outputs."""

    scored = compute_composite_scores(
        scored,
        metric_map=metric_map,
        composite_scale=composite_scale,
        entity_columns=entity_columns,
    )
    scored = assign_performance_category(scored, thresholds)
    scored = evaluate_improvement_flags(scored, improvement_rules)
    scored = add_rankings(scored, entity_columns)
    return scored


def _prepare_metric_map(config: Dict) -> Dict[str, object]:
    return build_metric_map(config)


def validate_input_schema(df: pd.DataFrame) -> None:
    """
    Validate that input DataFrame has required columns.
    
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ['entity_id', 'vendor_code', 'Week_Order', 'metric', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Input data missing required columns: {missing_cols}. "
            f"Expected columns: {required_cols}"
        )


def _generate_enhanced_outputs(
    config: Dict,
    vendor_scored: pd.DataFrame,
    asin_scored: pd.DataFrame,
    logger
) -> None:
    """
    Generate enhanced scoring outputs: scoring_matrix.parquet and vendor_scoreboard.parquet.
    
    This creates wide-format outputs with per-metric contributions for better explainability.
    """
    from .ingestion_utils import ensure_directory
    
    # Determine output directory
    output_dir = Path(config.get("paths", {}).get("scoring_output_dir", config["paths"]["processed_dir"]))
    ensure_directory(output_dir)
    
    # Build scoring matrix from vendor and ASIN scored data
    # Combine vendor and ASIN data
    all_scored = []
    if not vendor_scored.empty:
        df_v = vendor_scored.copy()
        df_v['entity_id'] = df_v['vendor_code']
        all_scored.append(df_v)
    if not asin_scored.empty:
        df_a = asin_scored.copy()
        df_a['entity_id'] = df_a.get('vendor_code', '') + '_' + df_a.get('asin', '')
        all_scored.append(df_a)
    
    if not all_scored:
        logger.warning("No scored data to generate enhanced outputs")
        return
    
    combined = pd.concat(all_scored, ignore_index=True)
    
    # Build wide format with contributions per metric
    # Get unique entity/week combinations
    if 'Week_Order' not in combined.columns and 'week_label' in combined.columns:
        # Map week labels to order
        combined['Week_Order'] = combined.groupby('week_label').ngroup() + 1
    elif 'Week_Order' not in combined.columns:
        combined['Week_Order'] = 1
    
    # Pivot metrics to wide format
    entity_cols = ['entity_id', 'vendor_code', 'Week_Order']
    
    # Get available entity columns
    if 'vendor_name' in combined.columns:
        entity_cols.append('vendor_name')
    
    # Build pivot with contributions
    pivot_data = []
    for entity_id in combined['entity_id'].unique():
        entity_data = combined[combined['entity_id'] == entity_id]
        
        # Get first row for entity info
        first_row = entity_data.iloc[0]
        row_dict = {
            'entity_id': entity_id,
            'vendor_code': first_row.get('vendor_code', ''),
            'Week_Order': first_row.get('Week_Order', 1)
        }
        
        if 'vendor_name' in entity_data.columns:
            row_dict['vendor_name'] = first_row.get('vendor_name', '')
        
        # Add composite score if available
        if 'Composite_Score' in entity_data.columns:
            row_dict['score_composite'] = first_row['Composite_Score']
            row_dict['score_final'] = first_row['Composite_Score']
        
        # Add per-metric data
        for _, metric_row in entity_data.iterrows():
            metric_name = metric_row.get('metric', '')
            if not metric_name:
                continue
            
            # Add standardized value
            if 'Normalized_Value' in metric_row:
                row_dict[f'std_{metric_name}'] = metric_row['Normalized_Value']
            
            # Add contribution
            if 'Weighted_Score' in metric_row:
                row_dict[f'contrib_{metric_name}'] = metric_row['Weighted_Score']
                
                # Compute percentage contribution
                if 'Composite_Score' in first_row and first_row['Composite_Score'] != 0:
                    pct = (metric_row['Weighted_Score'] / first_row['Composite_Score']) * 100
                    row_dict[f'pct_contrib_{metric_name}'] = pct
        
        # Add placeholder columns for enhanced features
        row_dict['penalty_total'] = 0.0
        row_dict['score_final_conf'] = 1.0
        row_dict['reliability'] = 1.0
        
        pivot_data.append(row_dict)
    
    scoring_matrix = pd.DataFrame(pivot_data)
    
    # Save scoring matrix
    try:
        matrix_path = output_dir / "scoring_matrix.parquet"
        scoring_matrix.to_parquet(matrix_path, index=False)
        logger.info(f"Saved scoring_matrix.parquet with {len(scoring_matrix)} rows")
    except Exception as e:
        logger.warning(f"Failed to save parquet, falling back to CSV: {e}")
        matrix_csv = output_dir / "scoring_matrix.csv"
        scoring_matrix.to_csv(matrix_csv, index=False)
        logger.info(f"Saved scoring_matrix.csv with {len(scoring_matrix)} rows")
    
    # Build vendor scoreboard
    vendor_scoreboard = scoring_matrix.groupby(['vendor_code', 'Week_Order']).agg({
        'score_final': 'mean'
    }).reset_index()
    
    # Add rank within week
    vendor_scoreboard['rank'] = vendor_scoreboard.groupby('Week_Order')['score_final'].rank(
        ascending=False, method='min'
    ).astype(int)
    
    # Save vendor scoreboard
    try:
        scoreboard_path = output_dir / "vendor_scoreboard.parquet"
        vendor_scoreboard.to_parquet(scoreboard_path, index=False)
        logger.info(f"Saved vendor_scoreboard.parquet with {len(vendor_scoreboard)} rows")
    except Exception as e:
        logger.warning(f"Failed to save parquet, falling back to CSV: {e}")
        scoreboard_csv = output_dir / "vendor_scoreboard.csv"
        vendor_scoreboard.to_csv(scoreboard_csv, index=False)
        logger.info(f"Saved vendor_scoreboard.csv with {len(vendor_scoreboard)} rows")


def run_phase2(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """Execute the scoring pipeline and return computed scorecards."""

    config = load_config(config_path)
    logger = setup_logging(config)

    logger.info("Starting Phase 2 scoring with configuration %s", config_path)
    normalized_df = load_normalized_input(config, logger=logger)
    if normalized_df.empty:
        logger.warning("Normalized dataset is empty. No scoring will be performed.")
        return {"vendor": pd.DataFrame(), "asin": pd.DataFrame(), "audit": pd.DataFrame()}

    scoring_config = config.get("scoring", {})
    fallback_normalised = float(scoring_config.get("fallback_normalized_value", 0.5))
    metric_map = _prepare_metric_map(config)
    actual_metric_map = metric_map
    aggregation = scoring_config.get("aggregation", "mean")
    composite_scale = float(scoring_config.get("composite_scale", 1000))
    thresholds = scoring_config.get("thresholds", {})
    improvement_rules = scoring_config.get("improvement_rules", {})

    vendor_scored, vendor_entity_columns = _prepare_entity_frame(
        normalized_df,
        metric_map=actual_metric_map,
        entity_type="vendor",
        aggregation=aggregation,
        fallback_normalised=fallback_normalised,
    )
    vendor_scored = _apply_scoring(
        vendor_scored,
        metric_map=actual_metric_map,
        composite_scale=composite_scale,
        thresholds=thresholds,
        improvement_rules=improvement_rules,
        entity_columns=vendor_entity_columns,
    )

    asin_scored, asin_entity_columns = _prepare_entity_frame(
        normalized_df,
        metric_map=actual_metric_map,
        entity_type="asin",
        aggregation=aggregation,
        fallback_normalised=fallback_normalised,
    )
    asin_scored = _apply_scoring(
        asin_scored,
        metric_map=actual_metric_map,
        composite_scale=composite_scale,
        thresholds=thresholds,
        improvement_rules=improvement_rules,
        entity_columns=asin_entity_columns,
    )

    audit_frames = []
    if not vendor_scored.empty:
        audit_frames.append(vendor_scored.assign(entity_level="vendor"))
    if not asin_scored.empty:
        audit_frames.append(asin_scored.assign(entity_level="asin"))
    audit_df = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()

    vendor_paths = persist_outputs(vendor_scored, config, "vendor", logger)
    asin_paths = persist_outputs(asin_scored, config, "asin", logger)
    audit_path = write_audit(audit_df, config)

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "timestamp": timestamp,
        "records_vendor": int(len(vendor_scored)),
        "records_asin": int(len(asin_scored)),
        "metric_weights": {name: config.get("scoring", {}).get("metric_weights", {}).get(name) for name in actual_metric_map},
        "top_vendor": vendor_scored.sort_values("Composite_Score", ascending=False)["vendor_code"].head(1).tolist(),
        "score_range": {
            "min": float(pd.concat([vendor_scored["Composite_Score"], asin_scored["Composite_Score"]]).min())
            if not vendor_scored.empty or not asin_scored.empty
            else None,
            "max": float(pd.concat([vendor_scored["Composite_Score"], asin_scored["Composite_Score"]]).max())
            if not vendor_scored.empty or not asin_scored.empty
            else None,
        },
        "output_paths": {
            "vendor_csv": str(vendor_paths[0]),
            "vendor_parquet": str(vendor_paths[1]) if vendor_paths[1] else None,
            "asin_csv": str(asin_paths[0]),
            "asin_parquet": str(asin_paths[1]) if asin_paths[1] else None,
            "audit_csv": str(audit_path),
        },
    }
    write_metadata(metadata, config)

    top_vendors, bottom_vendors = summarize_entities(vendor_scored, "vendor_code")
    if not top_vendors.empty:
        print("Top 5 vendors:\n", top_vendors.to_string(index=False))
    if not bottom_vendors.empty:
        print("Bottom 5 vendors:\n", bottom_vendors.to_string(index=False))

    logger.info("Phase 2 scoring completed for %s vendors and %s ASINs", metadata["records_vendor"], metadata["records_asin"])
    
    # Generate enhanced scoring matrix and vendor scoreboard
    try:
        _generate_enhanced_outputs(config, vendor_scored, asin_scored, logger)
    except Exception as e:
        logger.warning(f"Failed to generate enhanced outputs: {e}")
    
    return {"vendor": vendor_scored, "asin": asin_scored, "audit": audit_df}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Brightstar Phase 2 scoring")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    return parser


def main() -> None:
    """CLI entry point for Phase 2 scoring."""

    parser = build_arg_parser()
    args = parser.parse_args()
    run_phase2(config_path=args.config)


if __name__ == "__main__":
    main()
