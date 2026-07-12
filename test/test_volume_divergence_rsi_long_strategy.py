import importlib

import pandas as pd
import pytest
from pydantic import ValidationError

from strategy_engine import RiskIntent, SimulationPosition, StrategyBarContext


def _strategy_module():
    return importlib.import_module("strategies.volume_divergence_rsi_long")


def _config(**overrides):
    module = _strategy_module()
    values = {
        "ma_period": 3,
        "volume_lookback": 3,
        "macd_fast_period": 2,
        "macd_slow_period": 4,
        "divergence_lookback": 6,
        "divergence_valid_bars": 3,
        "rsi_period": 2,
    }
    return module.VolumeDivergenceRSILongConfig(**{**values, **overrides})


def _decision_frame(rows=12):
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    frame = pd.DataFrame(
        {
            "Open": [99.0] * rows,
            "High": [101.0] * rows,
            "Low": [98.0] * rows,
            "Close": [99.0] * rows,
            "Volume": [1_300_000.0] * rows,
            "ma_value": [100.0] * rows,
            "average_volume": [1_000_000.0] * rows,
            "volume_confirmed": [True] * rows,
            "macd_dif": [-1.0] * rows,
            "macd_dea": [-1.1] * rows,
            "rsi": [25.0] * rows,
            "bottom_divergence_recent": [True] * rows,
        },
        index=index,
    )
    frame.loc[index[-2], ["Close", "ma_value", "rsi"]] = [99.0, 100.0, 29.0]
    frame.loc[index[-1], ["Open", "High", "Low", "Close", "ma_value", "rsi"]] = [
        100.0,
        102.0,
        99.5,
        101.0,
        100.0,
        31.0,
    ]
    return frame


def test_strategy_module_exists_and_exports_dual_mode_definition():
    module = _strategy_module()

    assert module.STRATEGY_DEFINITION.strategy_id == "volume_divergence_rsi_long"
    assert module.STRATEGY_DEFINITION.supported_modes == (
        "single_stock",
        "signal_portfolio",
    )


def test_config_rejects_invalid_periods_and_profit_lock_order():
    module = _strategy_module()

    with pytest.raises(ValidationError, match="macd_fast_period"):
        module.VolumeDivergenceRSILongConfig(
            macd_fast_period=26,
            macd_slow_period=12,
        )
    with pytest.raises(ValidationError, match="locked_profit_pct"):
        module.VolumeDivergenceRSILongConfig(
            profit_activation_pct=5,
            locked_profit_pct=5,
        )


def test_macd_bottom_divergence_compares_lower_price_low_with_higher_dif_low():
    module = _strategy_module()
    close = pd.Series([14, 10, 13, 12, 11, 9], dtype="float64")
    dif = pd.Series([-1, -2, -1.2, -0.8, -0.5, -1], dtype="float64")

    assert module.has_macd_bottom_divergence(close, dif, lookback=6) is True
    assert module.has_macd_bottom_divergence(close, dif * -1, lookback=6) is False


def test_divergence_event_expires_without_being_refreshed_by_same_lows():
    module = _strategy_module()
    close = pd.Series([14, 10, 13, 12, 11, 9, 10, 11, 12, 13], dtype="float64")
    dif = pd.Series([-1, -2, -1.2, -0.8, -0.5, -1, -0.7, -0.5, -0.3, 0], dtype="float64")

    events = module.calculate_bottom_divergence_events(close, dif, lookback=6)
    recent = events.astype(int).rolling(3, min_periods=1).max().astype(bool)

    assert events.tolist() == [False, False, False, False, False, True, False, False, False, False]
    assert recent.tolist()[-5:] == [True, True, True, False, False]


def test_rsi_handles_one_direction_price_runs():
    module = _strategy_module()

    rising = module.calculate_rsi([1, 2, 3, 4, 5], 2)
    falling = module.calculate_rsi([5, 4, 3, 2, 1], 2)

    assert rising[-1] == 100
    assert falling[-1] == 0


def test_indicator_preparation_is_prefix_invariant():
    module = _strategy_module()
    index = pd.date_range("2025-01-01", periods=90, freq="D")
    close = pd.Series(
        [100 + row * 0.1 + (row % 12 - 6) * 0.4 for row in range(90)],
        index=index,
    )
    data = pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": [1_000_000 + row * 100 for row in range(90)],
        },
        index=index,
    )
    config = _config()
    prefix = module.STRATEGY_DEFINITION.prepare_frame(data.iloc[:70], config)
    changed = data.copy()
    changed.loc[index[70]:, ["Close", "Volume"]] = [10_000, 99_000_000]
    full = module.STRATEGY_DEFINITION.prepare_frame(changed, config).iloc[:70]

    pd.testing.assert_frame_equal(prefix, full)


def test_minimum_history_combines_macd_warmup_and_divergence_window():
    module = _strategy_module()
    config = _config(macd_slow_period=4, divergence_lookback=6)

    minimum = module.STRATEGY_DEFINITION.min_history_bars(config)

    assert minimum == 9


def test_strict_signal_emits_next_open_entry_with_ten_percent_size_and_stop():
    module = _strategy_module()
    frame = _decision_frame()
    config = _config()

    decision = module.STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", frame, len(frame) - 1, config)
    )

    assert decision.entry is not None
    assert decision.entry.order_type == "next_open"
    assert decision.entry.suggested_position_pct == 0.10
    assert decision.entry.metadata["stop_loss_pct"] == 3


@pytest.mark.parametrize(
    "mutation",
    [
        ("previous_close_above_ma",),
        ("current_close_below_ma",),
        ("volume_missing",),
        ("divergence_missing",),
        ("rsi_cross_missing",),
    ],
)
def test_entry_requires_every_confirmation(mutation):
    module = _strategy_module()
    frame = _decision_frame()
    name = mutation[0]
    if name == "previous_close_above_ma":
        frame.iloc[-2, frame.columns.get_loc("Close")] = 101
    elif name == "current_close_below_ma":
        frame.iloc[-1, frame.columns.get_loc("Close")] = 99
    elif name == "volume_missing":
        frame.iloc[-1, frame.columns.get_loc("volume_confirmed")] = False
    elif name == "divergence_missing":
        frame.iloc[-1, frame.columns.get_loc("bottom_divergence_recent")] = False
    else:
        frame.iloc[-2, frame.columns.get_loc("rsi")] = 31

    decision = module.STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", frame, len(frame) - 1, _config())
    )

    assert decision.entry is None


def test_profit_activation_locks_profit_and_trails_peak_without_loosening():
    module = _strategy_module()
    frame = _decision_frame()
    frame.iloc[-1, frame.columns.get_loc("High")] = 106
    position = SimulationPosition(
        "SH603019",
        100,
        "2026-01-01",
        100,
        holding_bars=5,
        risk=RiskIntent(stop_price=97),
    )

    activated = module.STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            frame,
            len(frame) - 1,
            _config(),
            position,
            {"peak_price": 104, "trend_weak_bars": 0},
        )
    )

    assert activated.risk_update is not None
    assert activated.risk_update.stop_price == pytest.approx(103.88)
    assert activated.next_state["peak_price"] == 106

    position_with_tighter_stop = SimulationPosition(
        **{**position.__dict__, "risk": RiskIntent(stop_price=109)}
    )
    unchanged = module.STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            frame,
            len(frame) - 1,
            _config(),
            position_with_tighter_stop,
            {"peak_price": 110, "trend_weak_bars": 0},
        )
    )
    assert unchanged.risk_update.stop_price == 109


def test_two_closes_below_ma_emit_trend_exit():
    module = _strategy_module()
    frame = _decision_frame()
    frame.iloc[-1, frame.columns.get_loc("Close")] = 99
    position = SimulationPosition("SH603019", 100, "2026-01-01", 100, holding_bars=5)

    decision = module.STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            frame,
            len(frame) - 1,
            _config(),
            position,
            {"peak_price": 102, "trend_weak_bars": 1},
        )
    )

    assert decision.exit is not None
    assert decision.exit.reason == "trend_break"


def test_twenty_unproductive_bars_emit_time_exit():
    module = _strategy_module()
    frame = _decision_frame()
    position = SimulationPosition("SH603019", 100, "2026-01-01", 100, holding_bars=20)

    decision = module.STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            frame,
            len(frame) - 1,
            _config(),
            position,
            {"peak_price": 102, "trend_weak_bars": 0},
        )
    )

    assert decision.exit is not None
    assert decision.exit.reason == "unproductive_time_exit"
