import importlib

import numpy as np
import pandas as pd
import pytest

from strategy_engine import SimulationPosition, StrategyBarContext
from strategy_metadata import get_strategy_metadata


def load_strategy_module():
    try:
        return importlib.import_module("strategies.boll_macd_breakout")
    except ModuleNotFoundError:
        pytest.fail("BOLL MACD breakout strategy module is missing")


def valid_entry_values(**overrides):
    values = {
        "previous_close": 100,
        "previous_middle": 98,
        "previous_upper": 101,
        "current_close": 103,
        "current_middle": 99,
        "current_upper": 102,
        "current_dif": 0.3,
        "current_dea": 0.25,
        "recent_macd_golden_cross": True,
    }
    return {**values, **overrides}


def test_entry_requires_rising_middle_upper_breakout_bullish_macd_and_recent_golden_cross():
    module = load_strategy_module()

    assert module.should_enter_boll_macd_breakout(**valid_entry_values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"current_middle": 98},
        {"previous_close": 102},
        {"current_dif": 0.2, "current_dea": 0.25},
        {"recent_macd_golden_cross": False},
    ],
)
def test_entry_is_blocked_when_any_required_condition_is_missing(overrides):
    module = load_strategy_module()

    assert not module.should_enter_boll_macd_breakout(**valid_entry_values(**overrides))


def test_recent_macd_golden_cross_accepts_cross_within_confirmation_window():
    module = load_strategy_module()

    assert module.has_recent_macd_golden_cross(
        dif_values=[-0.3, -0.2, 0.1, 0.2, 0.3, 0.4],
        dea_values=[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        confirmation_bars=5,
    )


def test_recent_macd_golden_cross_rejects_cross_older_than_confirmation_window():
    module = load_strategy_module()

    assert not module.has_recent_macd_golden_cross(
        dif_values=[-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        dea_values=[0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        confirmation_bars=5,
    )


def test_bollinger_upper_uses_population_standard_deviation():
    module = load_strategy_module()

    upper = module.bollinger_upper([1, 2, 3], period=3, stddev=2)

    assert upper[-1] == pytest.approx(2 + 2 * np.sqrt(2 / 3))


def test_risk_prices_use_configured_stop_loss_percentage():
    module = load_strategy_module()

    stop_price, take_price = module.get_boll_macd_risk_prices(
        entry_price=100,
        stop_loss_pct=1,
        take_profit_pct=3,
    )

    assert stop_price == pytest.approx(99)
    assert take_price == pytest.approx(103)


def test_risk_prices_use_configured_take_profit_percentage():
    module = load_strategy_module()

    stop_price, take_price = module.get_boll_macd_risk_prices(
        entry_price=100,
        stop_loss_pct=1.5,
        take_profit_pct=1.5,
    )

    assert stop_price == pytest.approx(98.5)
    assert take_price == pytest.approx(101.5)


@pytest.mark.parametrize("invalid_value", [0, 0.15, 10.1, np.inf, np.nan])
@pytest.mark.parametrize("name", ["stop_loss_pct", "take_profit_pct"])
def test_risk_percentages_must_follow_approved_range_and_step(name, invalid_value):
    module = load_strategy_module()
    values = {"entry_price": 100, "stop_loss_pct": 1, "take_profit_pct": 1}
    values[name] = invalid_value

    with pytest.raises(ValueError, match="0.1 increments"):
        module.get_boll_macd_risk_prices(**values)


def test_strategy_defaults_and_metadata_expose_approved_risk_grid():
    module = load_strategy_module()
    metadata = get_strategy_metadata("boll_macd_breakout")
    params = {param.name: param for param in metadata.parameters}
    config = module.BollMACDConfig()

    assert config.stop_loss_pct == 1
    assert config.take_profit_pct == 1
    assert config.macd_confirmation_bars == 5
    assert metadata.label == "BOLL+MACD上轨突破策略"
    assert params["stop_loss_pct"].default == 1
    assert params["stop_loss_pct"].search_values == [0.5, 1.0, 1.5, 2.0, 3.0]
    assert params["take_profit_pct"].default == 1
    assert params["take_profit_pct"].search_values == [0.5, 1.0, 1.5, 2.0, 3.0]
    assert params["macd_confirmation_bars"].default == 5
    assert params["macd_confirmation_bars"].search_values == [3, 5, 10]
    for name in ("stop_loss_pct", "take_profit_pct"):
        assert params[name].min_value == 0.1
        assert params[name].max_value == 10
        assert params[name].step == 0.1


def test_boll_definition_emits_entry_and_fixed_risk_exit():
    module = load_strategy_module()
    config = module.BollMACDConfig(
        boll_period=5,
        fast_period=2,
        slow_period=5,
        signal_period=2,
        macd_confirmation_bars=2,
    )
    frame = pd.DataFrame(
        {
            "Open": [100.0] * 9,
            "High": [101.0] * 9,
            "Low": [99.0] * 9,
            "Close": [100.0] * 8 + [103.0],
            "Volume": [1_000_000] * 9,
            "boll_middle": [98.0] * 8 + [99.0],
            "boll_upper": [101.0] * 8 + [102.0],
            "macd_dif": [-0.3, -0.2, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.3],
            "macd_dea": [-0.2, -0.15, -0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.25],
        },
        index=pd.date_range("2026-01-01", periods=9, freq="D"),
    )

    entry = module.STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", frame, 8, config)
    )
    exit_frame = frame.copy()
    exit_frame.loc[exit_frame.index[-1], "Close"] = 98.0
    exit_decision = module.STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            exit_frame,
            8,
            config,
            SimulationPosition("SH603019", 100, "2026-01-01", 100.0, holding_bars=2),
        )
    )

    assert entry.entry is not None
    assert entry.entry.order_type == "next_open"
    assert entry.entry.suggested_position_pct == 0.95
    assert exit_decision.exit is not None
    assert exit_decision.exit.reason == "stop_loss"


def test_boll_definition_preparation_is_prefix_invariant():
    module = load_strategy_module()
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    data = pd.DataFrame(
        {
            "Open": np.linspace(100, 120, 80),
            "High": np.linspace(101, 121, 80),
            "Low": np.linspace(99, 119, 80),
            "Close": np.linspace(100, 120, 80),
            "Volume": [1_000_000] * 80,
        },
        index=dates,
    )
    config = module.BollMACDConfig()
    prefix = module.STRATEGY_DEFINITION.prepare_frame(data.iloc[:60], config)
    changed = data.copy()
    changed.loc[dates[60]:, "Close"] = 10_000
    full = module.STRATEGY_DEFINITION.prepare_frame(changed, config).iloc[:60]

    pd.testing.assert_frame_equal(
        prefix[["boll_middle", "boll_upper", "macd_dif", "macd_dea"]],
        full[["boll_middle", "boll_upper", "macd_dif", "macd_dea"]],
    )
