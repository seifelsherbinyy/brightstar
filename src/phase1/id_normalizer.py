"""
ID normalization and canonicalization functions for Phase 1 data.

This module provides functions to normalize vendor IDs, ASINs, weeks,
and categories according to BrightStar canonical standards.
"""

import logging
import re
from typing import Any, Union

logger = logging.getLogger(__name__)


def normalize_asin(asin: Any) -> str:
    """
    Normalize ASIN to canonical format.

    Rules:
    - Convert to uppercase
    - Remove hyphens, spaces, and special characters
    - Keep only alphanumeric characters
    - Must be 10 characters (pad with zeros if shorter, truncate if longer)

    Args:
        asin: Raw ASIN value (can be string, number, or None)

    Returns:
        Normalized ASIN string in canonical format

    Raises:
        ValueError: If ASIN cannot be normalized
    """
    if asin is None or str(asin).strip() == "" or str(asin).lower() == "nan":
        raise ValueError("ASIN cannot be empty or null")

    # Convert to string and uppercase
    asin_str = str(asin).upper().strip()

    # Remove hyphens, spaces, and keep only alphanumeric
    asin_clean = re.sub(r"[^A-Z0-9]", "", asin_str)

    if not asin_clean:
        raise ValueError(f"ASIN '{asin}' has no valid characters after normalization")

    # Ensure 10 characters (standard ASIN length)
    if len(asin_clean) < 10:
        asin_clean = asin_clean.zfill(10)
    elif len(asin_clean) > 10:
        logger.warning(f"ASIN '{asin}' longer than 10 chars, truncating to first 10")
        asin_clean = asin_clean[:10]

    return asin_clean


def normalize_vendor_id(vendor_id: Any) -> str:
    """
    Normalize vendor ID to canonical format.

    Rules:
    - Convert to uppercase
    - Remove spaces and special characters (except underscores)
    - Keep only alphanumeric and underscores

    Args:
        vendor_id: Raw vendor ID value

    Returns:
        Normalized vendor ID string

    Raises:
        ValueError: If vendor ID cannot be normalized
    """
    if vendor_id is None or str(vendor_id).strip() == "" or str(vendor_id).lower() == "nan":
        raise ValueError("Vendor ID cannot be empty or null")

    # Convert to string and uppercase
    vendor_str = str(vendor_id).upper().strip()

    # Remove spaces and special chars, keep alphanumeric and underscores
    vendor_clean = re.sub(r"[^A-Z0-9_]", "", vendor_str)

    if not vendor_clean:
        raise ValueError(f"Vendor ID '{vendor_id}' has no valid characters after normalization")

    return vendor_clean


def normalize_week(week: Any) -> int:
    """
    Normalize week to canonical YYYYWW integer format.

    Rules:
    - Convert to integer
    - Must be in YYYYWW format (6 digits)
    - Year must be between 2020 and 2099
    - Week must be between 01 and 53

    Args:
        week: Raw week value (can be int, string, or float)

    Returns:
        Normalized week as integer in YYYYWW format

    Raises:
        ValueError: If week cannot be normalized or is invalid
    """
    if week is None or str(week).strip() == "" or str(week).lower() == "nan":
        raise ValueError("Week cannot be empty or null")

    try:
        # Convert to string, remove any non-digits
        week_str = str(week).strip()
        week_str = re.sub(r"[^0-9]", "", week_str)

        if not week_str:
            raise ValueError(f"Week '{week}' has no digits")

        # Convert to integer
        week_int = int(week_str)

        # Validate YYYYWW format (6 digits)
        if week_int < 100000 or week_int > 999999:
            raise ValueError(f"Week '{week}' must be 6 digits (YYYYWW format)")

        # Extract year and week components
        year = week_int // 100
        week_num = week_int % 100

        # Validate year range
        if year < 2020 or year > 2099:
            raise ValueError(f"Year {year} must be between 2020 and 2099")

        # Validate week number
        if week_num < 1 or week_num > 53:
            raise ValueError(f"Week number {week_num} must be between 01 and 53")

        return week_int

    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot normalize week '{week}': {str(e)}")


def normalize_category(category: Any) -> str:
    """
    Normalize category to canonical format.

    Rules:
    - Convert to title case
    - Strip leading/trailing whitespace
    - Normalize internal spacing

    Args:
        category: Raw category value

    Returns:
        Normalized category string in title case

    Raises:
        ValueError: If category cannot be normalized
    """
    if category is None or str(category).strip() == "" or str(category).lower() == "nan":
        raise ValueError("Category cannot be empty or null")

    # Convert to string, strip, and normalize spacing
    category_str = str(category).strip()
    category_str = re.sub(r"\s+", " ", category_str)

    # Convert to title case
    category_clean = category_str.title()

    if not category_clean:
        raise ValueError(f"Category '{category}' has no valid characters")

    return category_clean


def is_valid_asin(asin: str) -> bool:
    """
    Check if ASIN is in valid format after normalization.

    Args:
        asin: Normalized ASIN string

    Returns:
        True if valid, False otherwise
    """
    if not asin or len(asin) != 10:
        return False
    return bool(re.match(r"^[A-Z0-9]{10}$", asin))


def is_valid_vendor_id(vendor_id: str) -> bool:
    """
    Check if vendor ID is in valid format after normalization.

    Args:
        vendor_id: Normalized vendor ID string

    Returns:
        True if valid, False otherwise
    """
    if not vendor_id:
        return False
    return bool(re.match(r"^[A-Z0-9_]+$", vendor_id))


def is_valid_week(week: int) -> bool:
    """
    Check if week is in valid YYYYWW format.

    Args:
        week: Normalized week integer

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(week, int):
        return False

    year = week // 100
    week_num = week % 100

    return (
        202001 <= week <= 209952
        and 2020 <= year <= 2099
        and 1 <= week_num <= 53
    )
