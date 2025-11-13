"""Brightstar Phase 6: end-to-end pipeline orchestration."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .ingestion_utils import ensure_directory, load_config
from .phase1_ingestion import run_phase1
from .phase2_scoring import run_phase2
from .phase3_commentary import run_phase3
from .phase4_forecasting import run_phase4
from .phase5_dashboard import run_phase5

LOGGER_NAME = "brightstar.pipeline"


@dataclass
class PhaseResult:
    """Outcome details for a single pipeline phase."""

    name: str
    status: str
    detail: str
    duration_seconds: float
    output: Any = None


@dataclass
class PipelineRunResult:
    """Aggregate summary returned by :func:`run_pipeline`."""

    config_path: str
    started_at: datetime
    finished_at: datetime
    phases: List[PhaseResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when all executed phases succeeded or were skipped."""

        return all(phase.status in {"success", "skipped"} for phase in self.phases)


@dataclass
class PhaseDefinition:
    """Declarative description of an orchestrated phase."""

    name: str
    runner: Callable[[str], Any]
    description: str
    summariser: Callable[[Any], str]
    skip_check: Callable[[Dict[str, Any]], bool]


def _setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Create a pipeline logger writing to the configured logs directory."""

    log_dir = ensure_directory(config["paths"]["logs_dir"])
    file_name = config.get("logging", {}).get("pipeline_file_name", "brightstar_pipeline.log")
    level_name = config.get("logging", {}).get("pipeline_level", "INFO")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / file_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.debug("Pipeline logger initialised at %s", log_dir / file_name)
    return logger


def _validate_config(config: Dict[str, Any]) -> None:
    """Raise an informative error when key configuration blocks are missing."""

    required_sections = ["paths", "ingestion", "scoring", "commentary", "forecasting", "dashboard"]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Configuration missing sections: {', '.join(missing_sections)}")

    required_paths = ["raw_dir", "processed_dir", "logs_dir", "calendar_map"]
    paths_block = config["paths"]
    missing_paths = [key for key in required_paths if key not in paths_block]
    if missing_paths:
        raise ValueError(f"Configuration missing path entries: {', '.join(missing_paths)}")

    # Eagerly ensure important directories exist for idempotent operation.
    ensure_directory(Path(paths_block["processed_dir"]))
    ensure_directory(Path(paths_block["raw_dir"]))
    ensure_directory(Path(paths_block["logs_dir"]))


def _phase1_summary(output: Any) -> str:
    if isinstance(output, pd.DataFrame):
        return f"Normalised {len(output)} records." if not output.empty else "No records to normalise."
    return "Phase 1 produced no dataframe output."


def _phase2_summary(output: Any) -> str:
    if isinstance(output, dict):
        vendor = output.get("vendor")
        asin = output.get("asin")
        vendor_count = len(vendor) if isinstance(vendor, pd.DataFrame) else 0
        asin_count = len(asin) if isinstance(asin, pd.DataFrame) else 0
        return f"Scored {vendor_count} vendor rows and {asin_count} ASIN rows."
    return "Phase 2 produced no scorecard output."


def _phase3_summary(output: Any) -> str:
    if isinstance(output, pd.DataFrame):
        return f"Generated {len(output)} commentary rows."
    return "Phase 3 produced no commentary output."


def _phase4_summary(output: Any) -> str:
    if isinstance(output, dict):
        vendor = output.get("vendor")
        asin = output.get("asin")
        vendor_count = len(vendor) if isinstance(vendor, pd.DataFrame) else 0
        asin_count = len(asin) if isinstance(asin, pd.DataFrame) else 0
        return f"Forecasted {vendor_count} vendor rows and {asin_count} ASIN rows."
    return "Phase 4 produced no forecast output."


def _phase5_summary(output: Any) -> str:
    if isinstance(output, tuple) and len(output) == 2:
        path, summary = output
        if isinstance(path, Path):
            detail = f"Dashboard written to {path.name}"
            if isinstance(summary, dict):
                vendors = summary.get("total_vendors")
                avg_score = summary.get("average_score")
                freshness = summary.get("data_freshness")
                extra = ", ".join(
                    part
                    for part in [
                        f"vendors={vendors}" if vendors is not None else None,
                        f"avg_score={avg_score:.2f}" if isinstance(avg_score, (int, float)) else None,
                        f"freshness={freshness}" if freshness else None,
                    ]
                    if part
                )
                if extra:
                    detail = f"{detail} ({extra})"
            return detail
    return "Phase 5 produced no dashboard output."


def _output_exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()


def _commentary_exists(config: Dict[str, Any]) -> bool:
    commentary_dir = Path(config["paths"]["processed_dir"]) / "commentary"
    return (commentary_dir / "commentary.csv").exists()


def _forecast_exists(config: Dict[str, Any]) -> bool:
    paths = config["paths"]
    vendor = paths.get("forecast_vendor_csv")
    asin = paths.get("forecast_asin_csv")
    return _output_exists(vendor) and _output_exists(asin)


def _dashboard_exists(config: Dict[str, Any]) -> bool:
    dashboard_config = config.get("dashboard", {})
    directory = Path(dashboard_config.get("output_dir", config["paths"].get("dashboard_dir", "")))
    if not directory.exists():
        return False
    prefix = dashboard_config.get("file_prefix", "BrightStar_Dashboard")
    return any(path.name.startswith(prefix) and path.suffix.lower() == ".xlsx" for path in directory.glob("*.xlsx"))


PHASE_REGISTRY: Sequence[PhaseDefinition] = (
    PhaseDefinition(
        name="phase1",
        runner=run_phase1,
        description="Data ingestion & normalisation",
        summariser=_phase1_summary,
        skip_check=lambda cfg: _output_exists(cfg["paths"].get("normalized_output")),
    ),
    PhaseDefinition(
        name="phase2",
        runner=run_phase2,
        description="Scoring engine",
        summariser=_phase2_summary,
        skip_check=lambda cfg: _output_exists(cfg["paths"].get("scorecard_vendor_csv"))
        and _output_exists(cfg["paths"].get("scorecard_asin_csv")),
    ),
    PhaseDefinition(
        name="phase3",
        runner=run_phase3,
        description="Commentary engine",
        summariser=_phase3_summary,
        skip_check=_commentary_exists,
    ),
    PhaseDefinition(
        name="phase4",
        runner=run_phase4,
        description="Forecasting engine",
        summariser=_phase4_summary,
        skip_check=_forecast_exists,
    ),
    PhaseDefinition(
        name="phase5",
        runner=run_phase5,
        description="Dashboard generator",
        summariser=_phase5_summary,
        skip_check=_dashboard_exists,
    ),
)


def _select_phases(
    phases: Optional[Iterable[str]],
    start: Optional[str],
    end: Optional[str],
) -> List[str]:
    """Determine which phases should be executed based on CLI options."""

    available = [phase.name for phase in PHASE_REGISTRY]
    if phases:
        selected: List[str] = []
        for entry in phases:
            entry_normalised = entry.strip().lower()
            if entry_normalised in {"all", "*"}:
                return available
            if entry_normalised not in available:
                raise ValueError(f"Unknown phase '{entry}'. Available phases: {', '.join(available)}")
            if entry_normalised not in selected:
                selected.append(entry_normalised)
        return selected

    selected = available
    if start:
        if start not in available:
            raise ValueError(f"Unknown start phase '{start}'.")
        selected = available[available.index(start) :]
    if end:
        if end not in available:
            raise ValueError(f"Unknown end phase '{end}'.")
        end_index = available.index(end)
        selected = [phase for phase in selected if available.index(phase) <= end_index]
    return selected


def run_pipeline(
    config_path: str = "config.yaml",
    phases: Optional[Iterable[str]] = None,
    start_phase: Optional[str] = None,
    end_phase: Optional[str] = None,
    fail_fast: bool = False,
    resume: bool = False,
) -> PipelineRunResult:
    """Execute multiple Brightstar phases in sequence."""

    config = load_config(config_path)
    _validate_config(config)
    logger = _setup_logging(config)

    selected = _select_phases(phases, start_phase, end_phase)
    logger.info("Running pipeline for phases: %s", ", ".join(selected))

    results: List[PhaseResult] = []
    started = datetime.now(timezone.utc)

    for definition in PHASE_REGISTRY:
        if definition.name not in selected:
            continue

        if resume and definition.skip_check(config):
            detail = f"Skipping {definition.name}; existing outputs detected."
            results.append(
                PhaseResult(
                    name=definition.name,
                    status="skipped",
                    detail=detail,
                    duration_seconds=0.0,
                    output=None,
                )
            )
            logger.info(detail)
            continue

        logger.info("Starting %s (%s)", definition.name, definition.description)
        phase_start = time.perf_counter()
        try:
            output = definition.runner(config_path=config_path)
            duration = time.perf_counter() - phase_start
            detail = definition.summariser(output)
            results.append(
                PhaseResult(
                    name=definition.name,
                    status="success",
                    detail=detail,
                    duration_seconds=duration,
                    output=output,
                )
            )
            logger.info("%s completed in %.2fs: %s", definition.name, duration, detail)
        except Exception as exc:  # pragma: no cover - defensive logging path
            duration = time.perf_counter() - phase_start
            logger.exception("%s failed after %.2fs", definition.name, duration)
            results.append(
                PhaseResult(
                    name=definition.name,
                    status="failed",
                    detail=str(exc),
                    duration_seconds=duration,
                    output=None,
                )
            )
            if fail_fast:
                logger.error("Fail-fast enabled; stopping pipeline execution.")
                finished = datetime.now(timezone.utc)
                _print_summary(results)
                return PipelineRunResult(
                    config_path=config_path,
                    started_at=started,
                    finished_at=finished,
                    phases=results,
                )

    finished = datetime.now(timezone.utc)
    _print_summary(results)
    return PipelineRunResult(
        config_path=config_path,
        started_at=started,
        finished_at=finished,
        phases=results,
    )


def _print_summary(results: Sequence[PhaseResult]) -> None:
    if not results:
        print("No phases executed.")
        return

    lines = ["Brightstar pipeline summary:"]
    for result in results:
        lines.append(
            f"  - {result.name}: {result.status.upper()} ({result.detail})"
        )
    print("\n".join(lines))


def _split_phases_argument(argument: Optional[str]) -> Optional[List[str]]:
    if argument is None:
        return None
    return [part for part in argument.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the pipeline CLI."""

    parser = argparse.ArgumentParser(description="Brightstar end-to-end pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument(
        "--phases",
        help="Comma-separated list of phases to run (phase1,phase2,...) or 'all'",
    )
    parser.add_argument(
        "--start-phase",
        choices=[definition.name for definition in PHASE_REGISTRY],
        help="Execute phases starting from this phase onward",
    )
    parser.add_argument(
        "--end-phase",
        choices=[definition.name for definition in PHASE_REGISTRY],
        help="Stop execution after this phase",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop execution after the first failure")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases whose outputs already exist according to configuration paths",
    )
    return parser


def main() -> None:
    """CLI entry point for executing the Phase 6 pipeline."""

    parser = build_arg_parser()
    args = parser.parse_args()
    phase_list = _split_phases_argument(args.phases)
    run_pipeline(
        config_path=args.config,
        phases=phase_list,
        start_phase=args.start_phase,
        end_phase=args.end_phase,
        fail_fast=args.fail_fast,
        resume=args.resume,
    )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
