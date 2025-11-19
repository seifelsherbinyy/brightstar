"""
Phase 1: Schema-driven, auditable data loader for BrightStar v2.0 pipeline.

This module provides a canonical, schema-enforced loader that validates,
normalizes, and outputs audit-ready datasets for downstream phases.
"""

from .loader import load_phase1

__all__ = ["load_phase1"]
