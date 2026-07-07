from __future__ import annotations

from typing import Any

import pandas as pd

from portfolio_models import FactorConfig, SelectionConfig


FACTOR_KEYS = ("momentum", "volatility", "liquidity", "trend")


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
