import pandas as pd
from fastapi.testclient import TestClient

import main
from strategy_engine import SimulationPosition, StrategyBarContext
from strategies.ma_trend_risk_control import (
    MATrendConfig,
    STRATEGY_DEFINITION,
    get_ma_exit_reason,
    should_enter_ma_trend,
)


def test_ma_trend_enters_on_golden_cross_with_positive_momentum():
    assert should_enter_ma_trend(
        previous_fast_ma=10,
        previous_slow_ma=11,
        current_fast_ma=12,
        current_slow_ma=11,
        close=13,
        trend_ma_value=12,
        momentum_return=0.04,
    )


def test_ma_trend_exits_on_dead_cross():
    reason = get_ma_exit_reason(
        previous_fast_ma=12,
        previous_slow_ma=11,
        current_fast_ma=10,
        current_slow_ma=11,
        current_price=101,
        entry_price=100,
        holding_bars=5,
        stop_loss_pct=5,
        take_profit_pct=12,
        max_holding_bars=80,
    )

    assert reason == "dead_cross"


def test_ma_trend_filter_blocks_entry_below_trend_ma():
    assert not should_enter_ma_trend(
        previous_fast_ma=10,
        previous_slow_ma=11,
        current_fast_ma=12,
        current_slow_ma=11,
        close=11,
        trend_ma_value=12,
        momentum_return=0.04,
    )


def test_ma_trend_strategy_appears_in_strategy_list():
    main.load_strategy_modules()
    client = TestClient(main.app)

    response = client.get("/strategies")

    assert response.status_code == 200
    strategies = {item["name"]: item for item in response.json()}
    assert strategies["ma_trend_risk_control"]["display_name"] == "均线趋势风控策略"
    assert strategies["ma_trend_risk_control"]["parameters"]


def test_ma_trend_definition_emits_entry_and_dead_cross_exit():
    config = MATrendConfig(fast_ma=2, slow_ma=5, trend_ma=20, momentum_lookback=2)
    frame = pd.DataFrame(
        {
            "Open": [10.0] * 22,
            "High": [11.0] * 22,
            "Low": [9.0] * 22,
            "Close": [10.0] * 21 + [13.0],
            "Volume": [1_000_000] * 22,
            "fast_ma_value": [10.0] * 21 + [12.0],
            "slow_ma_value": [11.0] * 22,
            "trend_ma_value": [9.0] * 22,
            "momentum_return": [0.0] * 21 + [0.3],
        },
        index=pd.date_range("2026-01-01", periods=22, freq="D"),
    )
    entry = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", frame, 21, config)
    )
    exit_frame = frame.copy()
    exit_frame.loc[exit_frame.index[-2], ["fast_ma_value", "slow_ma_value"]] = [12, 11]
    exit_frame.loc[exit_frame.index[-1], ["fast_ma_value", "slow_ma_value"]] = [10, 11]
    exit_decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            exit_frame,
            21,
            config,
            SimulationPosition("SH603019", 100, "2026-01-01", 10.0, holding_bars=5),
        )
    )

    assert entry.entry is not None
    assert entry.entry.order_type == "next_open"
    assert exit_decision.exit is not None
    assert exit_decision.exit.reason == "dead_cross"


def test_ma_trend_preparation_is_prefix_invariant():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    data = pd.DataFrame(
        {
            "Open": range(100, 200),
            "High": range(101, 201),
            "Low": range(99, 199),
            "Close": range(100, 200),
            "Volume": [1_000_000] * 100,
        },
        index=dates,
    )
    config = MATrendConfig()
    prefix = STRATEGY_DEFINITION.prepare_frame(data.iloc[:80], config)
    changed = data.copy()
    changed.loc[dates[80]:, "Close"] = 10_000
    full = STRATEGY_DEFINITION.prepare_frame(changed, config).iloc[:80]

    pd.testing.assert_frame_equal(
        prefix[["fast_ma_value", "slow_ma_value", "trend_ma_value", "momentum_return"]],
        full[["fast_ma_value", "slow_ma_value", "trend_ma_value", "momentum_return"]],
    )
