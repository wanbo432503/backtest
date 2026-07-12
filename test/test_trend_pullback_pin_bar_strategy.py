import pandas as pd

from strategy_engine import SimulationPosition, StrategyBarContext
from strategies.trend_pullback_pin_bar import (
    STRATEGY_DEFINITION,
    TrendPullbackPinBarConfig,
)


def _pin_bar_frame(rows: int = 6) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Open": [10.7] * rows,
            "High": [10.9] * rows,
            "Low": [10.3] * rows,
            "Close": [10.8] * rows,
            "Volume": [2_000_000] * rows,
            "ma_short_value": [10.6] * rows,
            "ma_medium_value": [10.0] * rows,
            "ma_long_value": [9.5] * rows,
            "support": [10.3] * rows,
            "average_volume": [1_000_000] * rows,
            "atr": [0.4] * rows,
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="D"),
    )
    return frame


def test_pin_bar_definition_emits_one_bar_stop_entry_with_structural_risk():
    config = TrendPullbackPinBarConfig(
        short_ma_period=2,
        medium_ma_period=3,
        long_ma_period=4,
        support_lookback=2,
        volume_lookback=2,
        atr_period=2,
    )
    frame = _pin_bar_frame()

    decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", frame, len(frame) - 1, config)
    )

    assert decision.entry is not None
    assert decision.entry.order_type == "stop_next_bar"
    assert decision.entry.trigger_price == 10.9
    assert decision.entry.expires_after_bars == 1
    assert decision.entry.risk is not None
    assert decision.entry.risk.stop_price == 10.29
    assert decision.entry.risk.risk_budget_pct == 0.005


def test_pin_bar_definition_exits_after_two_weak_trend_closes():
    config = TrendPullbackPinBarConfig(
        short_ma_period=2,
        medium_ma_period=3,
        long_ma_period=4,
        support_lookback=2,
        volume_lookback=2,
        atr_period=2,
    )
    frame = _pin_bar_frame()
    frame.loc[frame.index[-1], "Close"] = 10.0
    position = SimulationPosition(
        "SH603019",
        100,
        "2026-01-01",
        10.9,
        holding_bars=3,
    )

    decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            frame,
            len(frame) - 1,
            config,
            position,
            {"trend_weak_bars": 1},
        )
    )

    assert decision.exit is not None
    assert decision.exit.reason == "trend_weak"
    assert decision.next_state == {"trend_weak_bars": 2}


def test_pin_bar_preparation_is_prefix_invariant():
    dates = pd.date_range("2025-01-01", periods=160, freq="D")
    data = pd.DataFrame(
        {
            "Open": range(100, 260),
            "High": range(101, 261),
            "Low": range(99, 259),
            "Close": range(100, 260),
            "Volume": range(1_000_000, 1_000_160),
        },
        index=dates,
    )
    config = TrendPullbackPinBarConfig()
    prefix = STRATEGY_DEFINITION.prepare_frame(data.iloc[:130], config)
    changed = data.copy()
    changed.loc[dates[130]:, ["Close", "Low", "Volume"]] = [10_000, 9_000, 99_000_000]
    full = STRATEGY_DEFINITION.prepare_frame(changed, config).iloc[:130]

    pd.testing.assert_frame_equal(
        prefix[["ma_short_value", "ma_medium_value", "ma_long_value", "support", "average_volume", "atr"]],
        full[["ma_short_value", "ma_medium_value", "ma_long_value", "support", "average_volume", "atr"]],
    )
