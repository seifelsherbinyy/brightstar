"""Utility helpers supporting Phase 4 forecasting workflows."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ingestion_utils import ensure_directory, load_config


LOGGER_NAME = "brightstar.phase4"


@dataclass
class ForecastContext:
    """Configuration bundle required to generate forecasts."""

    horizon: int
    min_history: int
    frequency: str
    metrics: Sequence[str]
    ensemble_weights: Dict[str, float]
    metric_directions: Dict[str, str]
    uncertainty_buffer: float


def setup_logging(config: Dict) -> logging.Logger:
    """Configure logging for the forecasting phase."""

    log_dir = ensure_directory(config["paths"]["logs_dir"])
    file_name = config.get("logging", {}).get("forecasting_file_name", "phase4_forecasting.log")
    log_path = log_dir / file_name

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, config.get("logging", {}).get("forecasting_level", "INFO")))
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

    logger.debug("Forecasting logger initialised at %s", log_path)
    return logger


def load_normalized_dataset(config: Dict, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load the normalized dataset created during Phase 1."""

    forecasting_config = config.get("forecasting", {})
    candidate_paths: List[Path] = []

    preferred = forecasting_config.get("normalized_input")
    if preferred:
        candidate_paths.append(Path(preferred))

    default_normalized = config["paths"].get("normalized_output")
    if default_normalized:
        candidate_paths.append(Path(default_normalized))

    candidate_paths.append(Path(config["paths"]["processed_dir"]) / "normalized_phase1.csv")

    for path in candidate_paths:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                try:
                    return pd.read_parquet(path)
                except Exception as exc:  # pragma: no cover - depends on parquet engines
                    if logger:
                        logger.warning(
                            "Unable to read parquet file %s (%s). Falling back to CSV if available.",
                            path,
                            exc,
                        )
                    continue
            if path.suffix.lower() in {".csv", ".txt"}:
                return pd.read_csv(path)

    raise FileNotFoundError(
        "Unable to locate normalized dataset. Ensure Phase 1 output exists before running Phase 4."
    )


def build_context(config: Dict) -> ForecastContext:
    """Assemble the forecasting context from configuration settings."""

    forecasting_config = config.get("forecasting", {})
    horizon = int(forecasting_config.get("horizon_weeks", 4))
    min_history = int(forecasting_config.get("min_history_weeks", 3))
    frequency = str(forecasting_config.get("frequency", "W-MON"))
    metrics = list(forecasting_config.get("metrics", []))
    if not metrics:
        raise ValueError("No metrics configured for forecasting.")

    ensemble_weights = forecasting_config.get("ensemble_weights", {})
    if not ensemble_weights:
        ensemble_weights = {"fallback": 1.0}

    metric_directions = forecasting_config.get("metric_directions") or config.get("scoring", {}).get(
        "metric_directions", {}
    )
    uncertainty_buffer = float(forecasting_config.get("uncertainty_buffer_pct", 0.15))

    return ForecastContext(
        horizon=horizon,
        min_history=min_history,
        frequency=frequency,
        metrics=metrics,
        ensemble_weights={str(k): float(v) for k, v in ensemble_weights.items()},
        metric_directions={str(k): str(v) for k, v in metric_directions.items()},
        uncertainty_buffer=uncertainty_buffer,
    )


def _prepare_series(df: pd.DataFrame, entity_columns: List[str]) -> pd.DataFrame:
    columns = entity_columns + ["metric", "week_start", "value"]
    return df[columns].copy()


def prepare_entity_frames(normalized_df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    """Aggregate normalized data into time series per entity and metric."""

    frame = normalized_df.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])  # ensure datetime handling

    if entity_type == "vendor":
        grouped = (
            frame.groupby(["vendor_code", "metric", "week_start"], as_index=False)["value"].sum()
        )
        grouped["entity_type"] = "vendor"
    else:
        grouped = (
            frame.groupby(["vendor_code", "asin", "metric", "week_start"], as_index=False)["value"].sum()
        )
        grouped["entity_type"] = "asin"

    return grouped.sort_values("week_start").reset_index(drop=True)


def _compute_future_index(last_week: pd.Timestamp, horizon: int, frequency: str) -> pd.DatetimeIndex:
    offset = pd.tseries.frequencies.to_offset(frequency)
    start = last_week + offset
    return pd.date_range(start=start, periods=horizon, freq=frequency)


def _prophet_forecast(history: pd.DataFrame, horizon: int, frequency: str, logger: Optional[logging.Logger]) -> Optional[pd.DataFrame]:
    try:  # pragma: no cover - prophet is optional in test environments
        from prophet import Prophet
    except Exception as exc:  # pragma: no cover - optional dependency
        if logger:
            logger.debug("Prophet unavailable (%s); skipping Prophet component.", exc)
        return None

    df = history.rename(columns={"week_start": "ds", "value": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq=frequency)
    forecast = model.predict(future).tail(horizon)
    result = pd.DataFrame(
        {
            "forecast_week": pd.to_datetime(forecast["ds"]),
            "forecast_value": forecast["yhat"],
            "lower_bound": forecast["yhat_lower"],
            "upper_bound": forecast["yhat_upper"],
        }
    )
    result["method"] = "prophet"
    return result


def _lstm_forecast(history: pd.DataFrame, horizon: int, frequency: str, logger: Optional[logging.Logger]) -> Optional[pd.DataFrame]:
    try:  # pragma: no cover - tensorflow is optional in test environments
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception as exc:  # pragma: no cover - optional dependency
        if logger:
            logger.debug("TensorFlow unavailable (%s); skipping LSTM component.", exc)
        return None

    values = history["value"].astype(float).to_numpy()
    if len(values) < 6:
        if logger:
            logger.debug("Insufficient history (%s points) for TensorFlow forecasting.", len(values))
        return None

    sequence_length = min(6, len(values))
    X, y = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i : i + sequence_length])
        y.append(values[i + sequence_length])

    if not X:
        if logger:
            logger.debug("Not enough sequence windows for LSTM model.")
        return None

    X = np.array(X).reshape((-1, sequence_length, 1))
    y = np.array(y)

    model = models.Sequential(
        [
            layers.Input(shape=(sequence_length, 1)),
            layers.LSTM(16, activation="tanh"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=100, verbose=0)

    history_values = values[-sequence_length:].tolist()
    predictions: List[float] = []
    for _ in range(horizon):
        input_arr = np.array(history_values[-sequence_length:]).reshape((1, sequence_length, 1))
        pred = float(model.predict(input_arr, verbose=0)[0][0])
        predictions.append(pred)
        history_values.append(pred)

    future_index = _compute_future_index(history["week_start"].iloc[-1], horizon, frequency)
    std = float(np.std(values)) if len(values) > 1 else 0.0
    result = pd.DataFrame(
        {
            "forecast_week": future_index,
            "forecast_value": predictions,
            "lower_bound": np.array(predictions) - std,
            "upper_bound": np.array(predictions) + std,
        }
    )
    result["method"] = "lstm"
    return result


def _baseline_forecast(history: pd.DataFrame, horizon: int, frequency: str, buffer_pct: float) -> pd.DataFrame:
    values = history["value"].astype(float).to_numpy()
    if len(values) >= 2:
        diffs = np.diff(values)
        trend = float(np.mean(diffs))
    else:
        trend = 0.0

    last_value = float(values[-1]) if len(values) else 0.0
    predictions = []
    current = last_value
    for _ in range(horizon):
        current = current + trend
        predictions.append(current)

    future_index = _compute_future_index(history["week_start"].iloc[-1], horizon, frequency)
    std = float(np.std(values)) if len(values) > 1 else abs(trend)
    buffer = [abs(val) * buffer_pct for val in predictions]
    lower = [val - std - buf for val, buf in zip(predictions, buffer)]
    upper = [val + std + buf for val, buf in zip(predictions, buffer)]
    result = pd.DataFrame(
        {
            "forecast_week": future_index,
            "forecast_value": predictions,
            "lower_bound": lower,
            "upper_bound": upper,
        }
    )
    result["method"] = "fallback"
    return result


def _combine_forecasts(
    forecasts: Dict[str, pd.DataFrame],
    future_index: pd.DatetimeIndex,
    weights: Dict[str, float],
) -> pd.DataFrame:
    if not forecasts:
        raise ValueError("At least one forecast method is required to combine results.")

    combined = pd.DataFrame({"forecast_week": future_index})
    total_weight = 0.0
    value_accumulator = np.zeros(len(future_index))
    lower_accumulator = np.zeros(len(future_index))
    upper_accumulator = np.zeros(len(future_index))

    for name, forecast in forecasts.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        aligned = (
            forecast.set_index("forecast_week").reindex(future_index).ffill().bfill()
        )
        value_accumulator += aligned["forecast_value"].to_numpy() * weight
        lower_accumulator += aligned["lower_bound"].to_numpy() * weight
        upper_accumulator += aligned["upper_bound"].to_numpy() * weight
        total_weight += weight

    if total_weight == 0:
        # fall back to equally weighted blend
        total_weight = float(len(forecasts))
        for forecast in forecasts.values():
            aligned = forecast.set_index("forecast_week").reindex(future_index).ffill().bfill()
            value_accumulator += aligned["forecast_value"].to_numpy()
            lower_accumulator += aligned["lower_bound"].to_numpy()
            upper_accumulator += aligned["upper_bound"].to_numpy()

    combined["forecast_value"] = value_accumulator / total_weight
    combined["lower_bound"] = lower_accumulator / total_weight
    combined["upper_bound"] = upper_accumulator / total_weight
    return combined


def _directional_potential_gain(
    last_value: float,
    forecast_value: float,
    direction: str,
) -> float:
    direction_normalized = str(direction).lower()
    if direction_normalized in {"down", "desc", "lower", "decrease"}:
        return max(last_value - forecast_value, 0.0)
    return max(forecast_value - last_value, 0.0)


def forecast_single_entity(
    history: pd.DataFrame,
    context: ForecastContext,
    metric: str,
    entity_descriptor: Dict[str, str],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Generate ensemble forecasts for a single entity/metric time series."""

    history = history.sort_values("week_start")
    history_weeks = len(history)
    last_week = history["week_start"].iloc[-1]
    future_index = _compute_future_index(last_week, context.horizon, context.frequency)

    status_flags: List[str] = []
    if history_weeks < context.min_history:
        status_flags.append("insufficient_history")

    forecasts: Dict[str, pd.DataFrame] = {}

    prophet_forecast = _prophet_forecast(history, context.horizon, context.frequency, logger)
    if prophet_forecast is not None:
        forecasts["prophet"] = prophet_forecast

    lstm_forecast = _lstm_forecast(history, context.horizon, context.frequency, logger)
    if lstm_forecast is not None:
        forecasts["lstm"] = lstm_forecast

    if not forecasts:
        status_flags.append("fallback")

    fallback_forecast = _baseline_forecast(history, context.horizon, context.frequency, context.uncertainty_buffer)
    forecasts.setdefault("fallback", fallback_forecast)

    combined = _combine_forecasts(forecasts, future_index, context.ensemble_weights)
    combined["methods"] = ",".join(sorted(forecasts.keys()))
    combined["status"] = ",".join(sorted(status_flags)) if status_flags else "ready"

    last_value = float(history["value"].iloc[-1])
    direction = context.metric_directions.get(metric, "up")

    combined["last_observed_week"] = last_week
    combined["last_observed_value"] = last_value
    combined["potential_gain"] = combined["forecast_value"].apply(
        lambda value: _directional_potential_gain(last_value, float(value), direction)
    )
    combined["metric"] = metric
    combined["history_weeks"] = history_weeks

    for key, value in entity_descriptor.items():
        combined[key] = value

    ordered_cols = [
        "entity_type",
        "vendor_code",
        "asin",
        "metric",
        "forecast_week",
        "forecast_value",
        "lower_bound",
        "upper_bound",
        "last_observed_week",
        "last_observed_value",
        "potential_gain",
        "history_weeks",
        "methods",
        "status",
    ]
    for column in ordered_cols:
        if column not in combined.columns:
            combined[column] = None

    return combined[ordered_cols]


def generate_forecasts(
    normalized_df: pd.DataFrame,
    context: ForecastContext,
    entity_type: str,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Generate forecasts for either vendor or ASIN entity levels."""

    if normalized_df.empty:
        return pd.DataFrame(
            columns=
            [
                "entity_type",
                "vendor_code",
                "asin",
                "metric",
                "forecast_week",
                "forecast_value",
                "lower_bound",
                "upper_bound",
                "last_observed_week",
                "last_observed_value",
                "potential_gain",
                "history_weeks",
                "methods",
                "status",
            ]
        )

    prepared = prepare_entity_frames(normalized_df, entity_type)

    entity_columns = ["vendor_code"] if entity_type == "vendor" else ["vendor_code", "asin"]
    results: List[pd.DataFrame] = []

    for metric in context.metrics:
        subset = prepared[prepared["metric"] == metric]
        if subset.empty:
            if logger:
                logger.debug("No data available for metric %s at entity level %s", metric, entity_type)
            continue

        group_columns = entity_columns
        for entity_keys, history in subset.groupby(group_columns):
            if not isinstance(entity_keys, tuple):
                entity_keys = (entity_keys,)
            descriptor = {"entity_type": entity_type, "vendor_code": entity_keys[0]}
            if entity_type == "asin":
                descriptor["asin"] = entity_keys[1]
            else:
                descriptor["asin"] = None

            forecast = forecast_single_entity(history, context, metric, descriptor, logger=logger)
            results.append(forecast)

    if not results:
        return pd.DataFrame(
            columns=
            [
                "entity_type",
                "vendor_code",
                "asin",
                "metric",
                "forecast_week",
                "forecast_value",
                "lower_bound",
                "upper_bound",
                "last_observed_week",
                "last_observed_value",
                "potential_gain",
                "history_weeks",
                "methods",
                "status",
            ]
        )

    return pd.concat(results, ignore_index=True)


def persist_forecast_outputs(df: pd.DataFrame, config: Dict, entity_type: str, logger: Optional[logging.Logger] = None) -> Tuple[Path, Optional[Path]]:
    """Write forecast outputs to CSV and Parquet if available."""

    paths_config = config.get("paths", {})
    ensure_directory(paths_config.get("forecasts_dir", "data/processed/forecasts"))

    if entity_type == "vendor":
        csv_path = Path(paths_config.get("forecast_vendor_csv", "data/processed/forecasts/vendor_forecasts.csv"))
        parquet_path = Path(
            paths_config.get("forecast_vendor_parquet", "data/processed/forecasts/vendor_forecasts.parquet")
        )
    else:
        csv_path = Path(paths_config.get("forecast_asin_csv", "data/processed/forecasts/asin_forecasts.csv"))
        parquet_path = Path(
            paths_config.get("forecast_asin_parquet", "data/processed/forecasts/asin_forecasts.parquet")
        )

    df_to_save = df.copy()
    df_to_save["forecast_week"] = df_to_save["forecast_week"].astype(str)
    df_to_save["last_observed_week"] = df_to_save["last_observed_week"].astype(str)

    df_to_save.to_csv(csv_path, index=False)

    parquet_written: Optional[Path] = None
    try:
        df_to_save.to_parquet(parquet_path, index=False)
        parquet_written = parquet_path
    except Exception as exc:  # pragma: no cover - depends on parquet engine availability
        if logger:
            logger.warning("Unable to write parquet output %s (%s).", parquet_path, exc)

    return csv_path, parquet_written


def write_metadata(metadata: Dict[str, object], config: Dict) -> Path:
    """Persist run metadata to disk."""

    paths_config = config.get("paths", {})
    metadata_path = Path(paths_config.get("forecast_metadata", "data/processed/forecasts/run_metadata.json"))
    ensure_directory(metadata_path.parent)
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, default=str)
    return metadata_path


def summarise_potential(df: pd.DataFrame, entity_column: str, top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top and bottom entities by potential gain."""

    if df.empty:
        empty = pd.DataFrame(columns=[entity_column, "potential_gain"])
        return empty, empty

    aggregated = df.groupby(entity_column)["potential_gain"].mean().reset_index()
    top = aggregated.sort_values("potential_gain", ascending=False).head(top_n)
    bottom = aggregated.sort_values("potential_gain", ascending=True).head(top_n)
    return top, bottom


def load_metadata(config_path: str) -> Dict:
    """Convenience wrapper used in tests to read metadata."""

    config = load_config(config_path)
    metadata_path = Path(config["paths"].get("forecast_metadata", "data/processed/forecasts/run_metadata.json"))
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as stream:
        return json.load(stream)
