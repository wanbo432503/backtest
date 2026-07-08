import pandas as pd

from factor_engine import calculate_strategy_factor_values
from portfolio_selection_strategy_library import get_selection_strategy
from test.fixtures.portfolio_ohlcv import build_ohlcv_frame


def test_breakout_strength_is_higher_for_new_high_breakout_series():
    strategy = get_selection_strategy("strong_trend_breakout")
    breakout = build_ohlcv_frame(periods=140, daily_return=0.0005)
    non_breakout = build_ohlcv_frame(periods=140, daily_return=0.0005)
    as_of_date = pd.Timestamp("2025-07-20")

    last_breakout_index = breakout.index[breakout.index < as_of_date][-1]
    prior_breakout_high = breakout.loc[breakout.index < last_breakout_index, "High"].tail(60).max()
    breakout.loc[last_breakout_index, "Close"] = prior_breakout_high * 1.06
    breakout.loc[last_breakout_index, "High"] = prior_breakout_high * 1.07

    last_flat_index = non_breakout.index[non_breakout.index < as_of_date][-1]
    prior_flat_high = non_breakout.loc[non_breakout.index < last_flat_index, "High"].tail(60).max()
    non_breakout.loc[last_flat_index, "Close"] = prior_flat_high * 0.97

    breakout_row = calculate_strategy_factor_values(
        breakout,
        as_of_date,
        strategy,
        min_history_bars=80,
    )
    non_breakout_row = calculate_strategy_factor_values(
        non_breakout,
        as_of_date,
        strategy,
        min_history_bars=80,
    )

    assert breakout_row["skip_reason"] is None
    assert breakout_row["factor_values"]["breakout_strength"] > 0
    assert breakout_row["factor_values"]["breakout_strength"] > non_breakout_row["factor_values"]["breakout_strength"]


def test_volume_expansion_is_higher_when_recent_volume_expands():
    strategy = get_selection_strategy("strong_trend_breakout")
    expanded = build_ohlcv_frame(periods=140, base_volume=1_000_000)
    stable = build_ohlcv_frame(periods=140, base_volume=1_000_000)
    as_of_date = pd.Timestamp("2025-07-20")
    included_index = expanded.index[expanded.index < as_of_date]

    expanded.loc[included_index[-8:], "Volume"] *= 4

    expanded_row = calculate_strategy_factor_values(
        expanded,
        as_of_date,
        strategy,
        min_history_bars=80,
    )
    stable_row = calculate_strategy_factor_values(
        stable,
        as_of_date,
        strategy,
        min_history_bars=80,
    )

    assert expanded_row["factor_values"]["volume_expansion"] > stable_row["factor_values"]["volume_expansion"]


def test_drawdown_factor_is_worse_for_deep_drop_series():
    strategy = get_selection_strategy("drawdown_control_rotation")
    smooth = build_ohlcv_frame(periods=140, daily_return=0.0008)
    deep_drop = build_ohlcv_frame(periods=140, daily_return=0.0008)
    as_of_date = pd.Timestamp("2025-07-20")
    included_index = deep_drop.index[deep_drop.index < as_of_date]

    deep_drop.loc[included_index[-20:-10], "Close"] *= 1.25
    deep_drop.loc[included_index[-10:], "Close"] *= 0.70

    smooth_row = calculate_strategy_factor_values(
        smooth,
        as_of_date,
        strategy,
        min_history_bars=80,
    )
    deep_drop_row = calculate_strategy_factor_values(
        deep_drop,
        as_of_date,
        strategy,
        min_history_bars=80,
    )

    assert deep_drop_row["factor_values"]["max_drawdown_window"] > smooth_row["factor_values"]["max_drawdown_window"]


def test_downside_volatility_is_lower_for_smooth_uptrend():
    strategy = get_selection_strategy("steady_low_vol_momentum")
    smooth = build_ohlcv_frame(periods=140, daily_return=0.001)
    jagged = build_ohlcv_frame(periods=140, daily_return=0.001)
    as_of_date = pd.Timestamp("2025-07-20")
    included_index = jagged.index[jagged.index < as_of_date]

    jagged.loc[included_index[-40::2], "Close"] *= 1.06
    jagged.loc[included_index[-39::2], "Close"] *= 0.94

    smooth_row = calculate_strategy_factor_values(
        smooth,
        as_of_date,
        strategy,
        min_history_bars=80,
    )
    jagged_row = calculate_strategy_factor_values(
        jagged,
        as_of_date,
        strategy,
        min_history_bars=80,
    )

    assert smooth_row["factor_values"]["downside_volatility"] < jagged_row["factor_values"]["downside_volatility"]


def test_strategy_factor_values_do_not_use_rebalance_date_bar_when_lookahead_safe():
    strategy = get_selection_strategy("steady_low_vol_momentum")
    data = build_ohlcv_frame(periods=180, daily_return=0.001)
    as_of_date = data.index[150]
    data.loc[as_of_date, "Close"] = data["Close"].iloc[149] * 3
    data.loc[as_of_date, "High"] = data.loc[as_of_date, "Close"] * 1.01

    safe = calculate_strategy_factor_values(
        data,
        as_of_date,
        strategy,
        min_history_bars=80,
        lookahead_safe=True,
    )
    unsafe = calculate_strategy_factor_values(
        data,
        as_of_date,
        strategy,
        min_history_bars=80,
        lookahead_safe=False,
    )

    assert unsafe["factor_values"]["momentum_return"] > safe["factor_values"]["momentum_return"] * 2


def test_strategy_factor_values_mark_insufficient_history():
    strategy = get_selection_strategy("strong_trend_breakout")
    data = build_ohlcv_frame(periods=20)

    row = calculate_strategy_factor_values(
        data,
        pd.Timestamp("2025-02-01"),
        strategy,
        min_history_bars=80,
    )

    assert row["skip_reason"] == "insufficient_history"
    assert row["factor_values"] == {}
