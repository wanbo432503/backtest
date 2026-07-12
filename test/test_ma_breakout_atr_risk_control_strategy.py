from fastapi.testclient import TestClient
import pandas as pd

import main
from strategies.ma_breakout_atr_risk_control import (
    MABreakoutATRConfig,
    STRATEGY_DEFINITION,
    calculate_atr_position_pct,
    get_ma_breakout_atr_exit_reason,
    should_enter_ma_breakout,
    should_enter_trend_bootstrap,
)
from strategy_engine import SimulationPosition, StrategyBarContext


def test_ma_breakout_atr_enters_on_trend_breakout_with_volume_confirmation():
    assert should_enter_ma_breakout(
        close=15.2,
        ma20=14.0,
        ma60=13.5,
        ma120=12.0,
        previous_highest_high=15.0,
        volume=1_600_000,
        average_volume=1_000_000,
        volume_multiplier=1.5,
    )


def test_ma_breakout_atr_blocks_entry_without_long_trend_or_volume():
    assert not should_enter_ma_breakout(
        close=15.2,
        ma20=13.0,
        ma60=13.5,
        ma120=12.0,
        previous_highest_high=15.0,
        volume=1_600_000,
        average_volume=1_000_000,
        volume_multiplier=1.5,
    )
    assert not should_enter_ma_breakout(
        close=15.2,
        ma20=14.0,
        ma60=13.5,
        ma120=12.0,
        previous_highest_high=15.0,
        volume=1_400_000,
        average_volume=1_000_000,
        volume_multiplier=1.5,
    )
    assert not should_enter_ma_breakout(
        close=15.2,
        ma20=14.0,
        ma60=13.5,
        ma120=12.0,
        previous_highest_high=15.0,
        volume=1_500_000,
        average_volume=1_000_000,
        volume_multiplier=1.5,
    )


def test_trend_bootstrap_allows_early_breakout_before_long_ma_is_ready():
    assert should_enter_trend_bootstrap(
        close=15.2,
        ma20=14.0,
        ma60=13.5,
        previous_highest_high=15.0,
        volume=1_600_000,
        average_volume=1_000_000,
        volume_multiplier=1.5,
    )
    assert not should_enter_trend_bootstrap(
        close=15.2,
        ma20=13.0,
        ma60=13.5,
        previous_highest_high=15.0,
        volume=1_600_000,
        average_volume=1_000_000,
        volume_multiplier=1.5,
    )


def test_ma_breakout_atr_exit_reasons_cover_ma_atr_and_late_weak_trend():
    assert get_ma_breakout_atr_exit_reason(
        close=13.8,
        ma20=14.0,
        ma60=13.5,
        highest_close=16.0,
        atr=0.6,
        atr_stop_multiplier=2.5,
        holding_bars=20,
        max_holding_bars=80,
    ) == "ma20_lost"

    assert get_ma_breakout_atr_exit_reason(
        close=14.4,
        ma20=14.0,
        ma60=13.5,
        highest_close=16.0,
        atr=0.6,
        atr_stop_multiplier=2.5,
        holding_bars=20,
        max_holding_bars=80,
    ) == "atr_trailing_stop"

    assert get_ma_breakout_atr_exit_reason(
        close=14.5,
        ma20=13.2,
        ma60=13.5,
        highest_close=15.0,
        atr=0.2,
        atr_stop_multiplier=2.5,
        holding_bars=81,
        max_holding_bars=80,
    ) == "late_weak_trend"


def test_ma_breakout_atr_position_size_decreases_as_atr_pct_rises():
    low_vol_position = calculate_atr_position_pct(
        close=100,
        atr=2,
        target_atr_risk_pct=0.02,
        min_position_pct=0.2,
        max_position_pct=0.95,
    )
    high_vol_position = calculate_atr_position_pct(
        close=100,
        atr=8,
        target_atr_risk_pct=0.02,
        min_position_pct=0.2,
        max_position_pct=0.95,
    )

    assert low_vol_position == 0.95
    assert high_vol_position == 0.25
    assert high_vol_position < low_vol_position


def test_ma_breakout_atr_strategy_appears_in_strategy_list():
    main.load_strategy_modules()
    client = TestClient(main.app)

    response = client.get("/strategies")

    assert response.status_code == 200
    strategies = {item["name"]: item for item in response.json()}
    strategy = strategies["ma_breakout_atr_risk_control"]
    assert strategy["display_name"] == "均线突破ATR风控策略"
    assert strategy["parameters"]


def test_ma_breakout_atr_definition_preserves_bootstrap_entry_and_atr_size():
    config = MABreakoutATRConfig(
        short_ma=5,
        medium_ma=20,
        long_ma=60,
        breakout_lookback=10,
        volume_lookback=5,
        bootstrap_bars=120,
        atr_period=5,
        target_atr_risk_pct=0.03,
        min_position_pct=0.2,
        max_position_pct=0.95,
    )
    frame = pd.DataFrame(
        {
            "Open": [13.0] * 22,
            "High": [14.0] * 21 + [15.4],
            "Low": [12.0] * 22,
            "Close": [13.0] * 21 + [15.2],
            "Volume": [1_000_000] * 21 + [1_600_000],
            "ma_short_value": [14.0] * 22,
            "ma_medium_value": [13.5] * 22,
            "ma_long_value": [float("nan")] * 22,
            "highest_high": [15.0] * 22,
            "average_volume": [1_000_000] * 22,
            "atr": [0.6] * 22,
        },
        index=pd.date_range("2026-01-01", periods=22, freq="D"),
    )

    decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", frame, 21, config)
    )

    assert decision.entry is not None
    assert decision.entry.order_type == "next_open"
    assert decision.entry.suggested_position_pct == 0.76


def test_ma_breakout_atr_definition_emits_atr_trailing_exit():
    config = MABreakoutATRConfig()
    rows = STRATEGY_DEFINITION.min_history_bars(config)
    frame = pd.DataFrame(
        {
            "Open": [14.4] * rows,
            "High": [15.0] * rows,
            "Low": [14.0] * rows,
            "Close": [14.4] * rows,
            "Volume": [1_000_000] * rows,
            "ma_short_value": [14.0] * rows,
            "ma_medium_value": [13.5] * rows,
            "ma_long_value": [12.0] * rows,
            "highest_high": [15.0] * rows,
            "average_volume": [1_000_000] * rows,
            "atr": [0.6] * rows,
        },
        index=pd.date_range("2025-01-01", periods=rows, freq="D"),
    )
    position = SimulationPosition(
        "SH603019",
        100,
        "2025-01-01",
        14.0,
        holding_bars=20,
        highest_price=16.0,
    )

    decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", frame, rows - 1, config, position)
    )

    assert decision.exit is not None
    assert decision.exit.reason == "atr_trailing_stop"


def test_ma_breakout_atr_preparation_is_prefix_invariant():
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
    config = MABreakoutATRConfig()
    prefix = STRATEGY_DEFINITION.prepare_frame(data.iloc[:130], config)
    changed = data.copy()
    changed.loc[dates[130]:, ["Close", "High", "Volume"]] = [10_000, 10_100, 99_000_000]
    full = STRATEGY_DEFINITION.prepare_frame(changed, config).iloc[:130]

    pd.testing.assert_frame_equal(
        prefix[["ma_short_value", "ma_medium_value", "ma_long_value", "highest_high", "average_volume", "atr"]],
        full[["ma_short_value", "ma_medium_value", "ma_long_value", "highest_high", "average_volume", "atr"]],
    )
