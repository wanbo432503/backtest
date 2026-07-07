import pandas as pd

from factor_engine import calculate_symbol_factors, score_candidates
from portfolio_models import FactorConfig, SelectionConfig
from test.fixtures.portfolio_ohlcv import build_ohlcv_frame


def test_score_candidates_ranks_upward_trend_above_downward_trend():
    data_by_symbol = {
        "SH603019": build_ohlcv_frame(daily_return=0.002),
        "SZ002241": build_ohlcv_frame(daily_return=-0.001),
    }

    rows = score_candidates(
        data_by_symbol,
        pd.Timestamp("2025-10-01"),
        FactorConfig(momentum_weight=1.0, volatility_weight=0, liquidity_weight=0, trend_weight=0),
        SelectionConfig(top_n=1, min_history_bars=120),
    )

    assert rows[0]["symbol"] == "SH603019"
    assert rows[0]["rank"] == 1
    assert rows[0]["score"] > rows[1]["score"]


def test_score_candidates_penalizes_high_volatility_symbol():
    stable = build_ohlcv_frame(daily_return=0.001)
    volatile = build_ohlcv_frame(daily_return=0.001)
    volatile.loc[volatile.index[::2], "Close"] *= 1.04
    volatile.loc[volatile.index[1::2], "Close"] *= 0.96

    rows = score_candidates(
        {"SH603019": stable, "SZ002241": volatile},
        pd.Timestamp("2025-10-01"),
        FactorConfig(momentum_weight=0, volatility_weight=-1.0, liquidity_weight=0, trend_weight=0),
        SelectionConfig(top_n=1, min_history_bars=120),
    )

    assert rows[0]["symbol"] == "SH603019"
    assert rows[0]["factor_values"]["volatility"] < rows[1]["factor_values"]["volatility"]


def test_score_candidates_rewards_liquid_symbol():
    liquid = build_ohlcv_frame(base_volume=2_000_000)
    illiquid = build_ohlcv_frame(base_volume=100_000)

    rows = score_candidates(
        {"SH603019": liquid, "SZ002241": illiquid},
        pd.Timestamp("2025-10-01"),
        FactorConfig(momentum_weight=0, volatility_weight=0, liquidity_weight=1.0, trend_weight=0),
        SelectionConfig(top_n=1, min_history_bars=120),
    )

    assert rows[0]["symbol"] == "SH603019"
    assert rows[0]["factor_values"]["liquidity"] > rows[1]["factor_values"]["liquidity"]


def test_calculate_symbol_factors_marks_insufficient_history():
    data = build_ohlcv_frame(periods=40)

    row = calculate_symbol_factors(
        data,
        pd.Timestamp("2025-03-01"),
        FactorConfig(momentum_lookback=60),
        min_history_bars=120,
    )

    assert row["skip_reason"] == "insufficient_history"


def test_score_candidates_does_not_use_trade_date_close_when_lookahead_safe():
    data = build_ohlcv_frame(daily_return=0.001)
    as_of_date = data.index[150]
    data.loc[as_of_date, "Close"] = data["Close"].iloc[149] * 3

    safe = calculate_symbol_factors(
        data,
        as_of_date,
        FactorConfig(momentum_lookback=20),
        min_history_bars=20,
        lookahead_safe=True,
    )
    unsafe = calculate_symbol_factors(
        data,
        as_of_date,
        FactorConfig(momentum_lookback=20),
        min_history_bars=20,
        lookahead_safe=False,
    )

    assert unsafe["factor_values"]["momentum"] > safe["factor_values"]["momentum"] * 2


def test_score_candidates_uses_symbol_tiebreaker_for_equal_scores():
    same = build_ohlcv_frame(daily_return=0.001)

    rows = score_candidates(
        {"SZ002241": same.copy(), "SH603019": same.copy()},
        pd.Timestamp("2025-10-01"),
        FactorConfig(momentum_weight=1.0, volatility_weight=0, liquidity_weight=0, trend_weight=0),
        SelectionConfig(top_n=1, min_history_bars=120),
    )

    assert [row["symbol"] for row in rows] == ["SH603019", "SZ002241"]
