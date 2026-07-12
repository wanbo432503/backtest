from dataclasses import replace

import pandas as pd

from optimization_models import AShareTradingConfig
from strategy_engine import SimulationPosition, StrategyBarContext
from strategy_simulator import SimulationConfig, run_strategy_simulation
from strategies.ma60_price_cross import (
    MA60PriceCrossConfig,
    STRATEGY_DEFINITION,
    count_ma_crosses,
    crossed_above_ma,
    crossed_below_ma,
)


def _frame(close: list[float], ma: list[float] | None = None) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-02", periods=len(close))
    values = [float(value) for value in close]
    frame = pd.DataFrame(
        {
            "Open": values,
            "High": [value + 0.2 for value in values],
            "Low": [value - 0.2 for value in values],
            "Close": values,
            "Volume": [1_000_000.0] * len(values),
        },
        index=index,
    )
    if ma is not None:
        frame["ma_value"] = ma
    return frame


def _position() -> SimulationPosition:
    return SimulationPosition(
        symbol="SH603019",
        shares=100,
        entry_date="2025-01-02",
        entry_price=10.0,
        holding_bars=5,
    )


def _filtered_signal_frame(
    *,
    before_previous=9.9,
    previous=10.1,
    current=10.3,
    atr=0.4,
    slope=0.02,
) -> pd.DataFrame:
    close = [9.0] * 77 + [before_previous, previous, current]
    frame = _frame(close, [10.0] * len(close))
    frame["atr_value"] = atr
    frame["ma_slope_return"] = slope
    return frame


def _trading() -> AShareTradingConfig:
    return AShareTradingConfig(
        t_plus_one=True,
        lot_size=100,
        limit_up_down_filter=False,
        volume_filter=False,
        slippage_pct=0,
        buy_commission_pct=0,
        sell_commission_pct=0,
        stamp_tax_pct=0,
        min_commission=0,
    )


def test_price_cross_helpers_use_strict_current_side():
    assert crossed_above_ma(10, 10, 10.1, 10)
    assert not crossed_above_ma(9, 10, 10, 10)
    assert crossed_below_ma(10, 10, 9.9, 10)
    assert not crossed_below_ma(11, 10, 10, 10)


def test_count_ma_crosses_counts_upward_and_downward_transitions():
    close = pd.Series([9.0, 11.0, 9.0, 11.0, 9.0])
    ma = pd.Series([10.0] * 5)

    assert count_ma_crosses(close, ma) == 4
    assert count_ma_crosses(close, ma, lookback_bars=2) == 2


def test_ma60_frame_prepares_atr_and_slope_filters():
    prepared = STRATEGY_DEFINITION.prepare_frame(
        _frame([float(value) for value in range(100, 200)]),
        MA60PriceCrossConfig(),
    )

    assert prepared["atr_value"].iloc[-1] > 0
    assert prepared["ma_slope_return"].iloc[-1] > 0


def test_entry_requires_second_day_to_clear_atr_upper_band():
    frame = _filtered_signal_frame()

    decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            symbol="SH603019",
            frame=frame,
            bar_index=len(frame) - 1,
            config=MA60PriceCrossConfig(),
        )
    )

    assert decision.entry is not None
    assert decision.entry.metadata["entry_threshold"] == 10.2


def test_entry_rejects_price_that_only_touches_ma60_buffer_zone():
    frame = _filtered_signal_frame(previous=9.9, current=10.1)

    decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            symbol="SH603019",
            frame=frame,
            bar_index=len(frame) - 1,
            config=MA60PriceCrossConfig(),
        )
    )

    assert decision.entry is None


def test_entry_rejects_non_rising_ma60():
    frame = _filtered_signal_frame(slope=0.005)

    decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            symbol="SH603019",
            frame=frame,
            bar_index=len(frame) - 1,
            config=MA60PriceCrossConfig(),
        )
    )

    assert decision.entry is None


def test_exit_uses_lower_atr_band_instead_of_raw_ma60_touch():
    inside_band = _filtered_signal_frame(current=9.95)
    below_band = _filtered_signal_frame(current=9.8)

    held_inside = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            inside_band,
            len(inside_band) - 1,
            MA60PriceCrossConfig(),
            position=_position(),
        )
    )
    held_below = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            below_band,
            len(below_band) - 1,
            MA60PriceCrossConfig(),
            position=_position(),
        )
    )

    assert held_inside.exit is None
    assert held_below.exit is not None


def test_entry_blocks_reentry_for_ten_trading_days_then_allows_fresh_breakout():
    frame = _filtered_signal_frame(previous=10.05, current=10.3)

    blocked = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            frame,
            len(frame) - 1,
            MA60PriceCrossConfig(),
            bars_since_exit=10,
        )
    )
    allowed = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            frame,
            len(frame) - 1,
            MA60PriceCrossConfig(),
            bars_since_exit=11,
        )
    )

    assert blocked.entry is None
    assert allowed.entry is not None


def test_reentry_accepts_persistent_trend_after_cooldown_and_slope_confirmation():
    frame = _filtered_signal_frame(previous=10.3, current=10.4)

    blocked = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH601288",
            frame,
            len(frame) - 1,
            MA60PriceCrossConfig(),
            bars_since_exit=10,
        )
    )
    continuation = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH601288",
            frame,
            len(frame) - 1,
            MA60PriceCrossConfig(),
            bars_since_exit=11,
        )
    )

    assert blocked.entry is None
    assert continuation.entry is not None
    assert continuation.entry.metadata["entry_mode"] == "trend_continuation"


def test_ma60_strategy_defaults_to_fifteen_percent_position():
    assert MA60PriceCrossConfig().position_pct == 0.15


def test_definition_emits_next_open_entry_with_hidden_cross_count_priority():
    close = [9.0] * 75 + [11.0, 9.0, 9.0, 11.0, 11.0]
    frame = _frame(close, [10.0] * len(close))
    frame["atr_value"] = 0.4
    frame["ma_slope_return"] = 0.02

    decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            symbol="SH603019",
            frame=frame,
            bar_index=len(frame) - 1,
            config=MA60PriceCrossConfig(),
        )
    )

    assert decision.entry is not None
    assert decision.entry.order_type == "next_open"
    assert decision.entry.strength == -3
    assert decision.entry.metadata["ma_cross_count"] == 3


def test_definition_emits_next_open_exit_only_for_held_position():
    frame = _filtered_signal_frame(previous=11.0, current=9.0)

    held = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            symbol="SH603019",
            frame=frame,
            bar_index=len(frame) - 1,
            config=MA60PriceCrossConfig(),
            position=_position(),
        )
    )
    flat = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            symbol="SH603019",
            frame=frame,
            bar_index=len(frame) - 1,
            config=MA60PriceCrossConfig(),
        )
    )

    assert held.exit is not None
    assert held.exit.reason == "price_below_ma60_atr_band"
    assert held.exit.order_type == "next_open"
    assert flat.exit is None


def test_ma60_preparation_is_prefix_invariant():
    data = _frame([float(value) for value in range(100, 200)])
    config = MA60PriceCrossConfig()
    prefix = STRATEGY_DEFINITION.prepare_frame(data.iloc[:80], config)
    changed = data.copy()
    changed.loc[changed.index[80]:, "Close"] = 10_000
    full = STRATEGY_DEFINITION.prepare_frame(changed, config).iloc[:80]

    pd.testing.assert_series_equal(prefix["ma_value"], full["ma_value"])


def test_signal_portfolio_buys_fewer_cross_stock_first():
    frequent = [9.0] * 70 + [
        11.0,
        9.0,
        11.0,
        9.0,
        11.0,
        9.0,
        11.0,
        9.0,
    ]
    stable = [9.0] * 78
    frequent_frame = _frame(frequent + [11.0, 11.0, 11.0], [10.0] * 81)
    stable_frame = _frame(stable + [11.0, 11.0, 11.0], [10.0] * 81)
    for frame in (frequent_frame, stable_frame):
        frame["atr_value"] = 0.4
        frame["ma_slope_return"] = 0.02
    definition = replace(
        STRATEGY_DEFINITION,
        prepare_frame=lambda data, config: data.copy(),
    )
    signal_date = stable_frame.index[-2]

    result = run_strategy_simulation(
        definition,
        MA60PriceCrossConfig(),
        {
            "SH603019": frequent_frame,
            "SZ002241": stable_frame,
        },
        SimulationConfig(
            initial_cash=100_000,
            max_positions=1,
            max_position_pct=1,
            target_gross_exposure=1,
            trading=_trading(),
            start_date=signal_date.strftime("%Y-%m-%d"),
            end_date=stable_frame.index[-1].strftime("%Y-%m-%d"),
        ),
    )

    buys = [trade for trade in result.trades if trade["side"] == "buy"]
    strengths = {event["symbol"]: event["strength"] for event in result.signal_events}
    assert strengths["SZ002241"] > strengths["SH603019"]
    assert [trade["symbol"] for trade in buys] == ["SZ002241"]
