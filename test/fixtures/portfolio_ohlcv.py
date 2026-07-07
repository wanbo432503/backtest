from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_PERIODS = 220


def build_demo_portfolio_request() -> dict[str, Any]:
    return {
        "start_date": DEFAULT_START_DATE,
        "end_date": DEFAULT_END_DATE,
        "initial_cash": 100000,
        "data_provider": "auto",
        "universe": {
            "symbols": ["SH603019", "SZ002241"],
            "max_symbols": 4,
            "allowed_code_prefixes": ["60", "00"],
        },
        "factors": {
            "momentum_lookback": 60,
            "volatility_lookback": 20,
            "liquidity_lookback": 20,
            "momentum_weight": 0.45,
            "volatility_weight": -0.25,
            "liquidity_weight": 0.20,
            "trend_weight": 0.10,
        },
        "selection": {
            "top_n": 1,
            "min_history_bars": 120,
        },
        "rebalance": {
            "frequency": "monthly",
            "weekday": 0,
            "monthday": 1,
            "lookahead_safe": True,
        },
        "risk": {
            "max_position_pct": 0.50,
            "target_gross_exposure": 0.95,
            "cash_buffer_pct": 0.05,
            "max_drawdown_stop_pct": 30,
        },
    }


def build_invalid_universe_symbols() -> list[str]:
    return ["SH603019", "SZ002241", "SZ300750", "SH688001", "SH600000"]


def build_portfolio_ohlcv_fixture(symbols: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    symbol_list = list(symbols or ["SH603019", "SZ002241"])
    frames: dict[str, pd.DataFrame] = {}

    for index, symbol in enumerate(symbol_list):
        if symbol == "SZ002241":
            frames[symbol] = build_ohlcv_frame(
                start_date=DEFAULT_START_DATE,
                periods=DEFAULT_PERIODS,
                base_price=18,
                daily_return=-0.0002,
                late_daily_return=0.003,
                switch_day=110,
                base_volume=1_400_000,
            )
        else:
            frames[symbol] = build_ohlcv_frame(
                start_date=DEFAULT_START_DATE,
                periods=DEFAULT_PERIODS,
                base_price=30 + index * 5,
                daily_return=0.001,
                late_daily_return=0.0004,
                switch_day=110,
                base_volume=1_000_000 + index * 120_000,
            )

    return frames


def build_ohlcv_frame(
    start_date: str = DEFAULT_START_DATE,
    periods: int = DEFAULT_PERIODS,
    base_price: float = 20,
    daily_return: float = 0.001,
    late_daily_return: float | None = None,
    switch_day: int | None = None,
    base_volume: int = 1_000_000,
    limit_up_days: Iterable[int] | None = None,
    limit_down_days: Iterable[int] | None = None,
) -> pd.DataFrame:
    limit_up_set = set(limit_up_days or [])
    limit_down_set = set(limit_down_days or [])
    dates = pd.bdate_range(start=start_date, periods=periods)

    closes = [float(base_price)]
    for day in range(1, periods):
        previous_close = closes[-1]
        if day in limit_up_set:
            close = previous_close * 1.101
        elif day in limit_down_set:
            close = previous_close * 0.899
        else:
            rate = daily_return
            if late_daily_return is not None and switch_day is not None and day >= switch_day:
                rate = late_daily_return
            close = previous_close * (1 + rate)
        closes.append(round(close, 4))

    opens = []
    highs = []
    lows = []
    volumes = []
    for day, close in enumerate(closes):
        previous_close = closes[day - 1] if day > 0 else close
        open_price = previous_close * (1 + (0.001 if day % 2 == 0 else -0.001))
        high = max(open_price, close) * 1.01
        low = min(open_price, close) * 0.99
        opens.append(round(open_price, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        volumes.append(int(base_volume + (day % 10) * 10_000))

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )
