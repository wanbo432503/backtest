import importlib

import numpy as np
import pandas as pd
import pytest
from backtesting import Backtest

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
        "previous_dif": 0.1,
        "previous_dea": 0.2,
        "current_close": 103,
        "current_middle": 99,
        "current_upper": 102,
        "current_dif": 0.3,
        "current_dea": 0.25,
    }
    return {**values, **overrides}


def test_entry_requires_rising_middle_upper_breakout_and_macd_golden_cross():
    module = load_strategy_module()

    assert module.should_enter_boll_macd_breakout(**valid_entry_values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"current_middle": 98},
        {"previous_close": 102},
        {"current_dif": 0.2, "current_dea": 0.25},
    ],
)
def test_entry_is_blocked_when_any_required_condition_is_missing(overrides):
    module = load_strategy_module()

    assert not module.should_enter_boll_macd_breakout(**valid_entry_values(**overrides))


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


def test_backtest_attaches_risk_to_actual_fill_after_entry_bar(monkeypatch):
    module = load_strategy_module()
    signals = iter([True, False, False])
    monkeypatch.setattr(
        module,
        "should_enter_boll_macd_breakout",
        lambda **kwargs: next(signals, False),
    )

    class FastBollMACDBreakoutStrategy(module.BollMACDBreakoutStrategy):
        boll_period = 2
        fast_period = 2
        slow_period = 3
        signal_period = 2

    data = pd.DataFrame(
        {
            "Open": [100] * 7 + [110, 110, 110],
            "High": [101] * 7 + [120, 112, 111],
            "Low": [99] * 7 + [90, 108, 109],
            "Close": [100] * 7 + [110, 110, 110],
            "Volume": [1_000_000] * 10,
        },
        index=pd.date_range("2026-01-01", periods=10, freq="B"),
    )

    stats = Backtest(
        data,
        FastBollMACDBreakoutStrategy,
        cash=100_000,
        commission=0,
        finalize_trades=True,
    ).run()
    trades = stats["_trades"]

    assert len(trades) == 1
    assert trades.iloc[0]["EntryPrice"] == pytest.approx(110)
    assert trades.iloc[0]["EntryBar"] == 7
    assert trades.iloc[0]["ExitBar"] == 8
    assert trades.iloc[0]["ExitPrice"] == pytest.approx(108.9)


def test_backtest_rejects_invalid_risk_before_any_entry_signal():
    module = load_strategy_module()
    data = pd.DataFrame(
        {
            "Open": [100] * 50,
            "High": [101] * 50,
            "Low": [99] * 50,
            "Close": [100] * 50,
            "Volume": [1_000_000] * 50,
        },
        index=pd.date_range("2026-01-01", periods=50, freq="B"),
    )

    backtest = Backtest(data, module.BollMACDBreakoutStrategy, cash=100_000)

    with pytest.raises(ValueError, match="0.1 increments"):
        backtest.run(stop_loss_pct=0.15)


def test_strategy_defaults_and_metadata_expose_approved_risk_grid():
    module = load_strategy_module()
    metadata = get_strategy_metadata("boll_macd_breakout")
    params = {param.name: param for param in metadata.parameters}

    assert module.BollMACDBreakoutStrategy.stop_loss_pct == 1
    assert module.BollMACDBreakoutStrategy.take_profit_pct == 1
    assert metadata.label == "BOLL+MACD上轨突破策略"
    assert params["stop_loss_pct"].default == 1
    assert params["stop_loss_pct"].search_values == [0.5, 1.0, 1.5, 2.0, 3.0]
    assert params["take_profit_pct"].default == 1
    assert params["take_profit_pct"].search_values == [0.5, 1.0, 1.5, 2.0, 3.0]
    for name in ("stop_loss_pct", "take_profit_pct"):
        assert params[name].min_value == 0.1
        assert params[name].max_value == 10
        assert params[name].step == 0.1
