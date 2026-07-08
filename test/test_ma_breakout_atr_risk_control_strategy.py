from fastapi.testclient import TestClient

import main
from strategies.ma_breakout_atr_risk_control import (
    calculate_atr_position_pct,
    get_ma_breakout_atr_exit_reason,
    should_enter_ma_breakout,
)


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
