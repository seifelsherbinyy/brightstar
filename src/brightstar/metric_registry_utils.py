"""Utilities for BrightStar metric registry management.

This module provides minimal, offline-compatible helpers to:
  - Load the metric registry
  - Detect unseen metrics from normalized Phase 1 output
  - Append pending metrics for manual curation
  - Canonicalize metrics using a predefined list + variant mapping

All functions are designed to be safe to call even when files are missing,
always returning a valid result without raising under normal conditions.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Sequence, Optional

import logging
import pandas as pd


BRIGHTSTAR_LOGGER = logging.getLogger("brightstar")


REGISTRY_COLUMNS = [
    "metric_name",
    "display_name",
    "unit",
    "direction",
    "category",
    "forecastable",
    # Extended descriptive fields (optional; filled during bootstrap)
    "description",
    "keywords",
]


# ============================================================
# Canonical metrics, variants, descriptions (Topic: Safe Canonical Registry)
# ============================================================

canonical_metrics: Sequence[str] = [
    "ASP",
    "Amzn_GV",
    "CCOGS_As_A_Percent_Of_Revenue",
    "CM",
    "CP",
    "CPPU",
    "Customer_Returns",
    "Deal_GMS",
    "Fill_Rate_Sourceable",
    "GV",
    "Net_Ordered_GMS",
    "Net_Ordered_Units",
    "Net_PPM",
    "Net_Shipped_Units",
    "On_Hand_Inventory_Units_Sellable",
    "PCOGS",
    "PPM",
    "Product_GMS",
    "SoROOS",
    "Total_Inventory_Units",
    "Vendor_Confirmation_Rate_Sourceable",
]

metric_variants: Dict[str, Sequence[str]] = {
    "ASP": ["ASP", "avg selling price", "average selling price"],
    "Amzn_GV": ["Amzn GV", "Amazon GV", "AB GV", "Amazon Business Glance Views"],
    "CCOGS_As_A_Percent_Of_Revenue": [
        "CCOGS As A % Of Revenue",
        "CCOGS % Revenue",
        "Contra COGS % of revenue",
        "CCOGS%Revenue",
    ],
    "CM": ["CM", "Contribution Margin", "contribution_margin"],
    "CP": ["CP($)", "CP", "Contribution Profit", "CP USD"],
    "CPPU": ["CPPU", "Contribution Profit Per Unit", "profit per unit"],
    "Customer_Returns": ["Customer Returns($)", "Customer Returns", "Returns ($)"],
    "Deal_GMS": ["Deal GMS($)", "Deal GMS", "Promo GMS"],
    "Fill_Rate_Sourceable": [
        "Fill Rate - Sourceable",
        "Fill_Rate_Sourceable",
        "Sourceable Fill Rate",
    ],
    "GV": ["GV", "Glance Views", "Glance View", "Detail Page Views"],
    "Net_Ordered_GMS": [
        "Net Ordered GMS($)",
        "Net Ordered GMS",
        "Net Ordered Gross Merchandise Sales",
    ],
    "Net_Ordered_Units": ["Net Ordered Units", "Ordered Units Net", "net_ordered_units"],
    "Net_PPM": ["Net PPM", "Net Pure Profit Margin"],
    "Net_Shipped_Units": ["Net Shipped Units", "Shipped Units", "net_shipped_units"],
    "On_Hand_Inventory_Units_Sellable": [
        "On-Hand Inventory Units (Sellable)",
        "Sellable On Hand Units",
        "On Hand Inventory Units Sellable",
    ],
    "PCOGS": ["PCOGS($)", "PCOGS", "Procurement COGS"],
    "PPM": ["PPM", "Procurement Profit Margin"],
    "Product_GMS": ["Product GMS($)", "Product GMS"],
    "SoROOS": ["SoROOS(%)", "SoROOS", "Sourceable Replenishment Out of Stock"],
    "Total_Inventory_Units": ["Total Inventory Units", "Inventory Units Total"],
    "Vendor_Confirmation_Rate_Sourceable": [
        "Vendor Confirmation Rate - Sourceable",
        "Vendor Confirmation Rate",
        "VCR Sourceable",
    ],
}

metric_descriptions: Dict[str, str] = {
    "ASP": (
        "Average selling price per shipped unit; shipped revenue divided by shipped units, "
        "used to track pricing strategy and portfolio value."
    ),
    "Amzn_GV": (
        "Amazon Business glance views; number of product detail page views from business "
        "customers, indicating B2B traffic, visibility and interest."
    ),
    "CCOGS_As_A_Percent_Of_Revenue": (
        "Share of customer revenue consumed by COGS and contra-COGS trade terms, showing "
        "how investments and discounts compress gross margin."
    ),
    "CM": (
        "Contribution margin percentage after shipping costs and trade terms, used by "
        "Amazon to assess vendor and product profitability versus targets."
    ),
    "CP": (
        "Contribution profit in absolute currency; contribution margin applied to shipped "
        "revenue, representing profit pool after direct costs and terms."
    ),
    "CPPU": (
        "Contribution profit per unit; contribution profit divided by shipped units, "
        "highlighting per-unit profitability versus price, costs and vendor funding."
    ),
    "Customer_Returns": (
        "Currency value of customer returns and refunds for shipped units, reducing net "
        "revenue and impacting sell-through and profitability diagnostics."
    ),
    "Deal_GMS": (
        "Gross merchandise sales revenue generated while products run on deals or "
        "promotions, capturing incremental uplift from temporary discounts."
    ),
    "Fill_Rate_Sourceable": (
        "Percentage of confirmed, sourceable purchase-order units actually received by "
        "Amazon, indicating vendor supply reliability for replenishable items."
    ),
    "GV": (
        "Glance views; number of times a product detail page is viewed when Amazon "
        "Retail is featured offer, measuring customer traffic."
    ),
    "Net_Ordered_GMS": (
        "Customer-ordered merchandise sales after cancellations, promotions and "
        "discounts, representing cleaned ordered revenue value for the selected period."
    ),
    "Net_Ordered_Units": (
        "Number of customer-ordered units net of cancellations, used with glance "
        "views to calculate conversion and understand demand."
    ),
    "Net_PPM": (
        "Net pure profit margin; Amazon’s retail margin after COGS, contra-COGS and "
        "discounts relative to shipped revenue, key negotiation metric."
    ),
    "Net_Shipped_Units": (
        "Units actually shipped to customers in the period, excluding cancellations, "
        "often used as demand proxy and for sell-through calculations."
    ),
    "On_Hand_Inventory_Units_Sellable": (
        "Quantity of sellable on-hand inventory units in Amazon fulfillment centers, "
        "excluding damaged or unsellable stock states."
    ),
    "PCOGS": (
        "Procurement cost of goods sold; Amazon’s effective cost for received units "
        "after vendor terms, used in PPM and Net PPM."
    ),
    "PPM": (
        "Procurement profit margin based on PCOGS and revenue before some contra-COGS "
        "adjustments; complements Net PPM in profitability analysis."
    ),
    "Product_GMS": (
        "Gross merchandise sales for specific products or ASINs, measuring shipped or "
        "ordered sales revenue before returns and allowances."
    ),
    "SoROOS": (
        "Sourceable replenishment out of stock; percentage of detail-page views where "
        "the item appears out of stock but still sourceable."
    ),
    "Total_Inventory_Units": (
        "Total units of inventory held by Amazon across sellable and unsellable "
        "statuses, summarizing overall stock exposure and risk."
    ),
    "Vendor_Confirmation_Rate_Sourceable": (
        "Percentage of Amazon purchase-order units the vendor confirms for sourceable "
        "ASINs, indicating supply capacity and willingness to meet demand."
    ),
}


def normalize_metric_key(raw: str) -> str:
    """Normalize raw metric text into a safe comparison token.

    - Lowercase
    - Remove $, %, (, )
    - Replace hyphens with space
    - Remove apostrophes
    - Replace spaces with underscores
    - Collapse repeated underscores
    """
    import re

    if raw is None:
        return ""
    s = str(raw).strip().lower()
    s = re.sub(r"[\$%\(\)]", "", s)
    s = s.replace("-", " ")
    s = s.replace("'", "")
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def match_to_canonical(raw_metric: str, variant_map: Dict[str, Sequence[str]]) -> Optional[str]:
    """Match an ingested metric header to a canonical registry key.

    Returns the canonical name or ``None`` if no match is found.
    """
    norm = normalize_metric_key(raw_metric)
    if not norm:
        return None
    for canonical, variants in (variant_map or {}).items():
        all_candidates = [canonical] + list(variants)
        for candidate in all_candidates:
            if norm == normalize_metric_key(candidate):
                return canonical
    return None


def bootstrap_predefined_metrics(registry_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all canonical metrics exist in the registry with descriptions.

    Adds rows for any missing canonical metric.
    Expected columns: metric_name, display_name, unit, direction,
                      category, forecastable, description, keywords
    """
    df = registry_df.copy() if isinstance(registry_df, pd.DataFrame) else pd.DataFrame()
    # Ensure required columns exist
    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    existing = set(map(str, df.get("metric_name", pd.Series(dtype=object)).dropna().unique().tolist()))
    to_add = [m for m in canonical_metrics if m not in existing]
    rows = []
    for canonical in to_add:
        desc = metric_descriptions.get(canonical, "")
        display = canonical.replace("_", " ")
        rows.append(
            {
                "metric_name": canonical,
                "display_name": display,
                "unit": "",
                "direction": "",
                "category": "",
                "forecastable": "",
                "description": desc,
                "keywords": desc.lower(),
            }
        )
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)[REGISTRY_COLUMNS]], ignore_index=True)
    # Drop exact duplicates on metric_name
    if "metric_name" in df.columns:
        df = df.drop_duplicates(subset=["metric_name"], keep="first")
    return df[REGISTRY_COLUMNS].copy()


def load_metric_registry(path: Path) -> pd.DataFrame:
    """Load the metric registry from ``path``.

    Behavior:
    - If the file exists, load it as a DataFrame (supports CSV/Parquet by extension).
    - If the file is missing or unreadable, return an empty DataFrame with the
      canonical registry columns.
    - Never raises; always returns a DataFrame.
    """

    try:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                try:
                    df = pd.read_parquet(path)
                except Exception as exc:  # pragma: no cover (engine/env specific)
                    BRIGHTSTAR_LOGGER.warning(
                        "Unable to read registry parquet at %s (%s). Falling back to empty registry.",
                        path,
                        exc,
                    )
                    return pd.DataFrame(columns=REGISTRY_COLUMNS)
            else:
                # default to CSV/TSV style reading
                df = pd.read_csv(path)

            # Normalize columns: ensure required structure even if file has extras/missing
            for c in REGISTRY_COLUMNS:
                if c not in df.columns:
                    df[c] = None
            # Keep only expected columns in defined order; do not bootstrap here
            df = df[REGISTRY_COLUMNS].copy()
            return df
    except Exception as exc:  # pragma: no cover
        BRIGHTSTAR_LOGGER.warning(
            "Error loading metric registry at %s (%s). Returning empty registry.", path, exc
        )

    # Missing or error → return empty structured DF (no bootstrap; tests expect empty when file missing)
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def detect_new_metrics(normalized_df: pd.DataFrame, registry_df: pd.DataFrame) -> List[str]:
    """Detect metrics present in ``normalized_df`` but absent from ``registry_df``.

    Returns a sorted list of unknown metric names.
    """

    if normalized_df is None or normalized_df.empty or "metric" not in normalized_df.columns:
        return []

    seen_metrics = set(map(str, normalized_df["metric"].dropna().unique().tolist()))
    if registry_df is None or registry_df.empty or "metric_name" not in registry_df.columns:
        registered = set()
    else:
        registered = set(map(str, registry_df["metric_name"].dropna().unique().tolist()))

    unknown = sorted(m for m in seen_metrics if m not in registered)
    return unknown


def append_pending_metrics(pending_path: Path, new_metrics: List[str]) -> None:
    """Append ``new_metrics`` to the pending metrics CSV at ``pending_path``.

    The CSV columns are defined by ``REGISTRY_COLUMNS``. Only the ``metric_name``
    is populated; all other metadata fields remain empty for manual completion.
    The function does not overwrite existing content and creates the file if
    missing. If ``new_metrics`` is empty, no file is created/modified.
    """

    if not new_metrics:
        return

    # Ensure directory exists
    pending_path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = pd.DataFrame({
        "metric_name": list(map(str, new_metrics)),
        "display_name": [None] * len(new_metrics),
        "unit": [None] * len(new_metrics),
        "direction": [None] * len(new_metrics),
        "category": [None] * len(new_metrics),
        "forecastable": [None] * len(new_metrics),
        "description": [None] * len(new_metrics),
        "keywords": [None] * len(new_metrics),
    })[REGISTRY_COLUMNS]

    if pending_path.exists():
        try:
            existing = pd.read_csv(pending_path)
            # Ensure structure
            for c in REGISTRY_COLUMNS:
                if c not in existing.columns:
                    existing[c] = None
            existing = existing[REGISTRY_COLUMNS]
            combined = pd.concat([existing, new_rows], ignore_index=True)
            # Do not deduplicate here to preserve append semantics; duplicates are acceptable.
            combined.to_csv(pending_path, index=False)
            return
        except Exception as exc:  # pragma: no cover
            BRIGHTSTAR_LOGGER.warning(
                "Unable to read existing pending metrics at %s (%s). Recreating file.",
                pending_path,
                exc,
            )

    # Create new file
    new_rows.to_csv(pending_path, index=False)
