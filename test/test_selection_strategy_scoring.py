import pandas as pd

from factor_engine import score_candidates, score_candidates_with_strategy
from portfolio_models import FactorConfig, SelectionConfig
from portfolio_selection_strategy_library import get_selection_strategy
from test.fixtures.portfolio_ohlcv import build_ohlcv_frame


def test_steady_low_vol_momentum_ranks_smooth_uptrend_above_jagged_high_return_series():
    strategy = get_selection_strategy("steady_low_vol_momentum")
    smooth = build_ohlcv_frame(periods=180, daily_return=0.0018, base_volume=1_400_000)
    jagged = build_ohlcv_frame(periods=180, daily_return=0.0025, base_volume=1_400_000)
    as_of_date = pd.Timestamp("2025-09-15")
    included_index = jagged.index[jagged.index < as_of_date]
    jagged.loc[included_index[-50::2], "Close"] *= 1.12
    jagged.loc[included_index[-49::2], "Close"] *= 0.88

    rows = score_candidates_with_strategy(
        {"SZ002241": jagged, "SH603019": smooth},
        as_of_date,
        SelectionConfig(top_n=1, min_history_bars=100),
        strategy,
    )

    assert rows[0]["symbol"] == "SH603019"
    assert rows[0]["strategy_id"] == "steady_low_vol_momentum"
    assert rows[0]["rank"] == 1
    assert rows[0]["normalized_strategy_factors"]["realized_volatility"] > rows[1]["normalized_strategy_factors"]["realized_volatility"]
    assert rows[0]["score"] > rows[1]["score"]


def test_strong_trend_breakout_ranks_breakout_with_volume_confirmation_above_non_breakout():
    strategy = get_selection_strategy("strong_trend_breakout")
    breakout = build_ohlcv_frame(periods=180, daily_return=0.001, base_volume=1_000_000)
    flat = build_ohlcv_frame(periods=180, daily_return=0.001, base_volume=1_000_000)
    as_of_date = pd.Timestamp("2025-09-15")
    breakout_index = breakout.index[breakout.index < as_of_date]
    last_index = breakout_index[-1]
    previous_high = breakout.loc[breakout.index < last_index, "High"].tail(60).max()
    breakout.loc[last_index, "Close"] = previous_high * 1.08
    breakout.loc[last_index, "High"] = previous_high * 1.09
    breakout.loc[breakout_index[-8:], "Volume"] *= 5

    rows = score_candidates_with_strategy(
        {"SH603019": breakout, "SZ002241": flat},
        as_of_date,
        SelectionConfig(top_n=1, min_history_bars=100),
        strategy,
    )

    assert rows[0]["symbol"] == "SH603019"
    assert rows[0]["normalized_strategy_factors"]["breakout_strength"] > rows[1]["normalized_strategy_factors"]["breakout_strength"]
    assert rows[0]["normalized_strategy_factors"]["volume_expansion"] > rows[1]["normalized_strategy_factors"]["volume_expansion"]


def test_drawdown_control_rotation_penalizes_deep_recent_drawdown():
    strategy = get_selection_strategy("drawdown_control_rotation")
    controlled = build_ohlcv_frame(periods=180, daily_return=0.0012, base_volume=1_200_000)
    deep_drop = build_ohlcv_frame(periods=180, daily_return=0.0012, base_volume=1_200_000)
    as_of_date = pd.Timestamp("2025-09-15")
    included_index = deep_drop.index[deep_drop.index < as_of_date]
    deep_drop.loc[included_index[-30:-12], "Close"] *= 1.30
    deep_drop.loc[included_index[-12:], "Close"] *= 0.68

    rows = score_candidates_with_strategy(
        {"SH603019": controlled, "SZ002241": deep_drop},
        as_of_date,
        SelectionConfig(top_n=1, min_history_bars=100),
        strategy,
    )

    assert rows[0]["symbol"] == "SH603019"
    assert rows[0]["normalized_strategy_factors"]["max_drawdown_window"] > rows[1]["normalized_strategy_factors"]["max_drawdown_window"]


def test_strategy_scoring_preserves_skip_reasons_and_symbol_tiebreaker():
    strategy = get_selection_strategy("steady_low_vol_momentum")
    same = build_ohlcv_frame(periods=180, daily_return=0.001)
    short = build_ohlcv_frame(periods=20, daily_return=0.001)

    rows = score_candidates_with_strategy(
        {
            "SZ002241": same.copy(),
            "SH603019": same.copy(),
            "SH600000": short,
        },
        pd.Timestamp("2025-09-15"),
        SelectionConfig(top_n=1, min_history_bars=100),
        strategy,
    )

    assert [row["symbol"] for row in rows[:2]] == ["SH603019", "SZ002241"]
    assert rows[-1]["symbol"] == "SH600000"
    assert rows[-1]["skip_reason"] == "insufficient_history"


def test_legacy_score_candidates_output_is_unchanged_for_factor_config_path():
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
    assert rows[0]["factor_values"].keys() == {"momentum", "volatility", "liquidity", "trend"}
    assert "strategy_id" not in rows[0]

