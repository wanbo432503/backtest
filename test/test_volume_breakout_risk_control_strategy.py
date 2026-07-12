import pandas as pd
from fastapi.testclient import TestClient

import main
from strategy_engine import SimulationPosition, StrategyBarContext
from strategies.volume_breakout_risk_control import (
    STRATEGY_DEFINITION,
    VolumeBreakoutConfig,
    should_enter_breakout,
)


def test_breakout_with_volume_enters():
    assert should_enter_breakout(
        close=12.5,
        previous_highest_close=12,
        volume=2500000,
        average_volume=1000000,
        volume_multiplier=2,
        previous_close=11.8,
        limit_up_down_filter=True,
    )


def test_breakout_without_enough_volume_does_not_enter():
    assert not should_enter_breakout(
        close=12.5,
        previous_highest_close=12,
        volume=1500000,
        average_volume=1000000,
        volume_multiplier=2,
        previous_close=11.8,
        limit_up_down_filter=True,
    )


def test_limit_up_filter_blocks_entry():
    assert not should_enter_breakout(
        close=11,
        previous_highest_close=10.5,
        volume=2500000,
        average_volume=1000000,
        volume_multiplier=2,
        previous_close=10,
        limit_up_down_filter=True,
    )


def test_volume_breakout_strategy_appears_in_strategy_list():
    main.load_strategy_modules()
    client = TestClient(main.app)

    response = client.get("/strategies")

    assert response.status_code == 200
    strategies = {item["name"]: item for item in response.json()}
    assert strategies["volume_breakout_risk_control"]["display_name"] == "放量突破风控策略"
    assert strategies["volume_breakout_risk_control"]["parameters"]


def test_volume_breakout_definition_emits_entry_and_breakout_line_exit():
    config = VolumeBreakoutConfig(breakout_lookback=3, volume_lookback=3)
    frame = pd.DataFrame(
        {
            "Open": [10.0] * 5,
            "High": [11.0] * 5,
            "Low": [9.0] * 5,
            "Close": [10.0, 10.5, 11.0, 11.8, 12.5],
            "Volume": [1_000_000] * 4 + [2_500_000],
            "highest_close": [10.0, 10.5, 11.0, 12.0, 12.5],
            "average_volume": [1_000_000] * 5,
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )
    entry = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", frame, 4, config)
    )
    exit_frame = frame.copy()
    exit_frame.loc[exit_frame.index[-1], "Close"] = 11.5
    exit_decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext(
            "SH603019",
            exit_frame,
            4,
            config,
            SimulationPosition("SH603019", 100, "2026-01-01", 12.0, holding_bars=3),
            {"breakout_line": 12.0},
        )
    )

    assert entry.entry is not None
    assert entry.next_state == {"breakout_line": 12.0}
    assert exit_decision.exit is not None
    assert exit_decision.exit.reason == "breakout_line_lost"


def test_volume_breakout_preparation_is_prefix_invariant():
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    data = pd.DataFrame(
        {
            "Open": range(100, 180),
            "High": range(101, 181),
            "Low": range(99, 179),
            "Close": range(100, 180),
            "Volume": range(1_000_000, 1_000_080),
        },
        index=dates,
    )
    config = VolumeBreakoutConfig()
    prefix = STRATEGY_DEFINITION.prepare_frame(data.iloc[:60], config)
    changed = data.copy()
    changed.loc[dates[60]:, ["Close", "Volume"]] = [10_000, 99_000_000]
    full = STRATEGY_DEFINITION.prepare_frame(changed, config).iloc[:60]

    pd.testing.assert_frame_equal(
        prefix[["highest_close", "average_volume"]],
        full[["highest_close", "average_volume"]],
    )
