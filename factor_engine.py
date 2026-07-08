from __future__ import annotations

from typing import Any

import pandas as pd

from portfolio_models import FactorConfig, SelectionConfig
from portfolio_selection_strategy_models import PortfolioSelectionStrategyDefinition


FACTOR_KEYS = ("momentum", "volatility", "liquidity", "trend")
FUNDAMENTAL_FACTOR_KEYS = {
    "pe_inverse",
    "pb_inverse",
    "roe",
    "revenue_growth",
    "profit_growth",
}


def calculate_symbol_factors(
    data: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    config: FactorConfig,
    min_history_bars: int = 120,
    lookahead_safe: bool = True,
) -> dict[str, Any]:
    history = _history_until(data, as_of_date, lookahead_safe)
    required_bars = max(
        min_history_bars,
        config.momentum_lookback + 1,
        config.volatility_lookback + 1,
        config.liquidity_lookback,
    )
    if len(history) < required_bars:
        return {
            "factor_values": {},
            "skip_reason": "insufficient_history",
        }

    close = history["Close"].astype(float)
    volume = history["Volume"].astype(float)
    returns = close.pct_change().dropna()

    momentum = close.iloc[-1] / close.iloc[-config.momentum_lookback - 1] - 1
    volatility = returns.tail(config.volatility_lookback).std()
    liquidity = (close * volume).tail(config.liquidity_lookback).mean()
    moving_average = close.tail(config.momentum_lookback).mean()
    trend = close.iloc[-1] / moving_average - 1 if moving_average else 0

    return {
        "factor_values": {
            "momentum": float(momentum),
            "volatility": float(volatility) if pd.notna(volatility) else 0.0,
            "liquidity": float(liquidity),
            "trend": float(trend),
        },
        "skip_reason": None,
    }


def calculate_strategy_factor_values(
    data: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    strategy: PortfolioSelectionStrategyDefinition,
    min_history_bars: int = 120,
    lookahead_safe: bool = True,
    parameter_overrides: dict[str, Any] | None = None,
    fundamentals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = _history_until(data, as_of_date, lookahead_safe)
    lookbacks = _strategy_lookbacks(strategy, parameter_overrides or {})
    required_bars = max([min_history_bars, *(lookback + 1 for lookback in lookbacks.values())])
    if len(history) < required_bars:
        return {
            "factor_values": {},
            "skip_reason": "insufficient_history",
        }

    close = history["Close"].astype(float)
    high = history["High"].astype(float) if "High" in history else close
    volume = history["Volume"].astype(float)
    returns = close.pct_change().dropna()
    factor_values: dict[str, float] = {}
    warnings: list[str] = []
    missing_fundamental_keys: list[str] = []

    for factor in strategy.factors:
        key = factor.key
        lookback = lookbacks.get(key)
        if key == "momentum_return" and lookback is not None:
            factor_values[key] = _momentum_return(close, lookback)
        elif key == "realized_volatility" and lookback is not None:
            factor_values[key] = _realized_volatility(returns, lookback)
        elif key == "downside_volatility" and lookback is not None:
            factor_values[key] = _downside_volatility(returns, lookback)
        elif key == "liquidity_turnover" and lookback is not None:
            factor_values[key] = _liquidity_turnover(close, volume, lookback)
        elif key == "ma_trend" and lookback is not None:
            factor_values[key] = _ma_trend(close, lookback)
        elif key == "breakout_strength" and lookback is not None:
            factor_values[key] = _breakout_strength(close, high, lookback)
        elif key == "volume_expansion" and lookback is not None:
            factor_values[key] = _volume_expansion(volume, lookback)
        elif key == "volume_stability" and lookback is not None:
            factor_values[key] = _volume_stability(volume, lookback)
        elif key == "max_drawdown_window" and lookback is not None:
            factor_values[key] = _max_drawdown_window(close, lookback)
        elif key == "recovery_strength" and lookback is not None:
            factor_values[key] = _recovery_strength(close, lookback)
        elif fundamentals and key in fundamentals:
            factor_values[key] = _safe_float(fundamentals.get(key))
        elif key in FUNDAMENTAL_FACTOR_KEYS:
            missing_fundamental_keys.append(key)

    if missing_fundamental_keys:
        warnings.append("missing_fundamentals")

    return {
        "factor_values": factor_values,
        "skip_reason": None,
        "warnings": warnings,
        "missing_fundamental_keys": missing_fundamental_keys,
    }


def score_candidates(
    data_by_symbol: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp | str,
    factor_config: FactorConfig,
    selection_config: SelectionConfig,
) -> list[dict[str, Any]]:
    rows = []
    for symbol, data in data_by_symbol.items():
        factor_row = calculate_symbol_factors(
            data,
            as_of_date,
            factor_config,
            min_history_bars=selection_config.min_history_bars,
            lookahead_safe=True,
        )
        rows.append({
            "symbol": symbol,
            "factor_values": factor_row["factor_values"],
            "skip_reason": factor_row["skip_reason"],
            "score": None,
            "rank": None,
        })

    scored_rows = [row for row in rows if row["skip_reason"] is None]
    normalized = _normalize_factors(scored_rows)
    weights = {
        "momentum": factor_config.momentum_weight,
        "volatility": factor_config.volatility_weight,
        "liquidity": factor_config.liquidity_weight,
        "trend": factor_config.trend_weight,
    }

    for row in scored_rows:
        row["normalized_factors"] = normalized[row["symbol"]]
        row["score"] = float(
            sum(row["normalized_factors"].get(key, 0.0) * weight for key, weight in weights.items())
        )

    rows.sort(
        key=lambda row: (
            row["skip_reason"] is not None,
            -(row["score"] if row["score"] is not None else float("-inf")),
            row["symbol"],
        )
    )

    rank = 1
    for row in rows:
        if row["skip_reason"] is None:
            row["rank"] = rank
            rank += 1

    return rows


def score_candidates_with_strategy(
    data_by_symbol: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp | str,
    selection_config: SelectionConfig,
    strategy: PortfolioSelectionStrategyDefinition,
    parameter_overrides: dict[str, Any] | None = None,
    fundamentals_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    overrides = parameter_overrides or {}
    rows = []
    for symbol, data in data_by_symbol.items():
        factor_row = calculate_strategy_factor_values(
            data,
            as_of_date,
            strategy,
            min_history_bars=selection_config.min_history_bars,
            lookahead_safe=True,
            parameter_overrides=overrides,
            fundamentals=(fundamentals_by_symbol or {}).get(symbol),
        )
        rows.append({
            "symbol": symbol,
            "strategy_id": strategy.strategy_id,
            "strategy_factor_values": factor_row["factor_values"],
            "factor_values": factor_row["factor_values"],
            "skip_reason": factor_row["skip_reason"],
            "warnings": list(factor_row.get("warnings", [])),
            "missing_fundamental_keys": list(factor_row.get("missing_fundamental_keys", [])),
            "score": None,
            "rank": None,
        })

    scored_rows = [row for row in rows if row["skip_reason"] is None]
    normalized = _normalize_strategy_factors(scored_rows, strategy)
    weights = _strategy_weights(strategy, overrides)

    for row in scored_rows:
        row["normalized_strategy_factors"] = normalized[row["symbol"]]
        row["normalized_factors"] = normalized[row["symbol"]]
        row["score"] = float(
            sum(
                row["normalized_strategy_factors"].get(key, 0.0) * weight
                for key, weight in weights.items()
            )
        )

    rows.sort(
        key=lambda row: (
            row["skip_reason"] is not None,
            -(row["score"] if row["score"] is not None else float("-inf")),
            row["symbol"],
        )
    )

    rank = 1
    for row in rows:
        if row["skip_reason"] is None:
            row["rank"] = rank
            rank += 1

    return rows


def _strategy_lookbacks(
    strategy: PortfolioSelectionStrategyDefinition,
    parameter_overrides: dict[str, Any],
) -> dict[str, int]:
    lookbacks: dict[str, int] = {}
    for factor in strategy.factors:
        override = parameter_overrides.get(factor.key, {})
        override_lookback = override.get("lookback") if isinstance(override, dict) else None
        if override_lookback is not None:
            lookbacks[factor.key] = int(override_lookback)
        elif factor.default_lookback is not None:
            lookbacks[factor.key] = factor.default_lookback
    return lookbacks


def _strategy_weights(
    strategy: PortfolioSelectionStrategyDefinition,
    parameter_overrides: dict[str, Any],
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for factor in strategy.factors:
        override = parameter_overrides.get(factor.key, {})
        override_weight = override.get("weight") if isinstance(override, dict) else None
        weights[factor.key] = (
            float(override_weight)
            if override_weight is not None
            else float(factor.default_weight)
        )
    return weights


def _history_until(
    data: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    lookahead_safe: bool,
) -> pd.DataFrame:
    frame = data.sort_index()
    timestamp = pd.Timestamp(as_of_date)
    if lookahead_safe:
        return frame[frame.index < timestamp]
    return frame[frame.index <= timestamp]


def _momentum_return(close: pd.Series, lookback: int) -> float:
    return _safe_float(close.iloc[-1] / close.iloc[-lookback - 1] - 1)


def _realized_volatility(returns: pd.Series, lookback: int) -> float:
    return _safe_float(returns.tail(lookback).std())


def _downside_volatility(returns: pd.Series, lookback: int) -> float:
    downside = returns.tail(lookback)
    downside = downside[downside < 0]
    if downside.empty:
        return 0.0
    return _safe_float(downside.std())


def _liquidity_turnover(close: pd.Series, volume: pd.Series, lookback: int) -> float:
    return _safe_float((close * volume).tail(lookback).mean())


def _ma_trend(close: pd.Series, lookback: int) -> float:
    moving_average = close.tail(lookback).mean()
    if not moving_average:
        return 0.0
    return _safe_float(close.iloc[-1] / moving_average - 1)


def _breakout_strength(close: pd.Series, high: pd.Series, lookback: int) -> float:
    previous_high = high.iloc[-lookback - 1 : -1].max()
    if not previous_high:
        return 0.0
    return _safe_float(close.iloc[-1] / previous_high - 1)


def _volume_expansion(volume: pd.Series, lookback: int) -> float:
    window = volume.tail(lookback)
    recent_count = max(1, min(5, len(window) // 3))
    recent = window.tail(recent_count).mean()
    baseline = window.iloc[: -recent_count].mean() if len(window) > recent_count else window.mean()
    if not baseline:
        return 0.0
    return _safe_float(recent / baseline - 1)


def _volume_stability(volume: pd.Series, lookback: int) -> float:
    window = volume.tail(lookback)
    mean_value = window.mean()
    if not mean_value:
        return 0.0
    coefficient = window.std() / mean_value
    return _safe_float(1 / (1 + coefficient))


def _max_drawdown_window(close: pd.Series, lookback: int) -> float:
    window = close.tail(lookback)
    running_peak = window.cummax()
    drawdown = window / running_peak - 1
    return abs(_safe_float(drawdown.min()))


def _recovery_strength(close: pd.Series, lookback: int) -> float:
    window = close.tail(lookback)
    trough = window.min()
    if not trough:
        return 0.0
    return _safe_float(close.iloc[-1] / trough - 1)


def _safe_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _normalize_strategy_factors(
    rows: list[dict[str, Any]],
    strategy: PortfolioSelectionStrategyDefinition,
) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {row["symbol"]: {} for row in rows}
    for factor in strategy.factors:
        key = factor.key
        values = [float(row["strategy_factor_values"].get(key, 0.0)) for row in rows]
        if not values:
            continue
        min_value = min(values)
        max_value = max(values)
        for row, value in zip(rows, values):
            if max_value == min_value:
                normalized[row["symbol"]][key] = 0.0
            elif factor.direction == "lower_better":
                normalized[row["symbol"]][key] = (max_value - value) / (max_value - min_value)
            else:
                normalized[row["symbol"]][key] = (value - min_value) / (max_value - min_value)
    return normalized


def _normalize_factors(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {row["symbol"]: {} for row in rows}
    for key in FACTOR_KEYS:
        values = [float(row["factor_values"].get(key, 0.0)) for row in rows]
        if not values:
            continue
        min_value = min(values)
        max_value = max(values)
        for row, value in zip(rows, values):
            if max_value == min_value:
                normalized[row["symbol"]][key] = 0.0
            else:
                normalized[row["symbol"]][key] = (value - min_value) / (max_value - min_value)
    return normalized
