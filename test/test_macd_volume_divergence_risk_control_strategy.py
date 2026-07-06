from fastapi.testclient import TestClient

import main
from strategies.macd_volume_divergence_risk_control import (
    get_macd_volume_exit_reason,
    has_bullish_macd_divergence,
    has_volume_confirmation,
    is_golden_cross,
    should_enter_continuation,
    should_enter_macd_volume,
)


def test_bullish_macd_divergence_detects_price_lower_low_with_indicator_higher_low():
    assert has_bullish_macd_divergence(
        closes=[12, 11, 10, 10.5, 9.8, 9.4],
        dif_values=[-1.1, -1.3, -1.5, -0.9, -0.8, -0.7],
        histogram_values=[-0.6, -0.8, -1.0, -0.5, -0.4, -0.3],
    )


def test_bullish_macd_divergence_accepts_histogram_higher_low_even_if_dif_lags():
    assert has_bullish_macd_divergence(
        closes=[12, 11, 10, 10.5, 9.8, 9.4],
        dif_values=[-1.1, -1.2, -1.3, -1.4, -1.45, -1.5],
        histogram_values=[-0.6, -0.8, -1.0, -0.5, -0.4, -0.3],
    )


def test_volume_confirmation_requires_enough_volume():
    assert has_volume_confirmation(volume=2200000, average_volume=1000000, multiplier=2.0)
    assert not has_volume_confirmation(volume=1500000, average_volume=1000000, multiplier=2.0)


def test_macd_volume_entry_accepts_divergence_golden_cross_with_volume():
    assert should_enter_macd_volume(
        previous_dif=-0.8,
        previous_dea=-0.7,
        current_dif=-0.45,
        current_dea=-0.5,
        current_close=20,
        volume=2600000,
        average_volume=1000000,
        volume_multiplier=2.0,
        divergence_detected=True,
        zero_axis_threshold=0.05,
    )


def test_macd_volume_entry_accepts_zero_axis_golden_cross_with_volume():
    assert should_enter_macd_volume(
        previous_dif=-0.05,
        previous_dea=-0.04,
        current_dif=0.03,
        current_dea=0.02,
        current_close=20,
        volume=1800000,
        average_volume=1000000,
        volume_multiplier=1.5,
        divergence_detected=False,
        zero_axis_threshold=0.01,
    )


def test_macd_volume_entry_blocks_underwater_cross_without_divergence():
    assert not should_enter_macd_volume(
        previous_dif=-1.2,
        previous_dea=-1.1,
        current_dif=-0.9,
        current_dea=-1.0,
        current_close=20,
        volume=3000000,
        average_volume=1000000,
        volume_multiplier=2.0,
        divergence_detected=False,
        zero_axis_threshold=0.01,
    )


def test_macd_volume_continuation_entry_accepts_intact_water_trend_near_trend_ma():
    assert should_enter_continuation(
        current_dif=1.2,
        current_dea=0.9,
        current_close=21,
        trend_ma_value=20,
        volume=1300000,
        average_volume=1000000,
        continuation_volume_multiplier=1.2,
        continuation_pullback_pct=8,
    )


def test_macd_volume_continuation_entry_blocks_overextended_price():
    assert not should_enter_continuation(
        current_dif=1.2,
        current_dea=0.9,
        current_close=25,
        trend_ma_value=20,
        volume=1300000,
        average_volume=1000000,
        continuation_volume_multiplier=1.2,
        continuation_pullback_pct=8,
    )


def test_macd_volume_exit_detects_dead_cross_before_large_loss():
    assert (
        get_macd_volume_exit_reason(
            previous_dif=0.8,
            previous_dea=0.6,
            current_dif=0.4,
            current_dea=0.5,
            recent_histogram=[0.5, 0.4, 0.3],
            current_price=101,
            entry_price=100,
            highest_price=108,
            trend_ma_value=95,
            holding_bars=8,
            histogram_fade_bars=3,
            stop_loss_pct=5,
            take_profit_pct=12,
            trailing_stop_pct=6,
            max_holding_bars=80,
        )
        == "dead_cross"
    )


def test_macd_volume_exit_holds_when_histogram_fades_but_trend_is_intact():
    reason = get_macd_volume_exit_reason(
        previous_dif=1.4,
        previous_dea=1.0,
        current_dif=1.2,
        current_dea=1.1,
        recent_histogram=[0.6, 0.4, 0.2],
        current_price=118,
        entry_price=100,
        highest_price=122,
        trend_ma_value=105,
        holding_bars=20,
        histogram_fade_bars=3,
        stop_loss_pct=5,
        take_profit_pct=12,
        trailing_stop_pct=8,
        max_holding_bars=80,
    )

    assert reason is None


def test_macd_volume_exit_holds_fixed_profit_while_trend_is_intact():
    reason = get_macd_volume_exit_reason(
        previous_dif=1.4,
        previous_dea=1.0,
        current_dif=1.5,
        current_dea=1.1,
        recent_histogram=[0.3, 0.4, 0.5],
        current_price=125,
        entry_price=100,
        highest_price=126,
        trend_ma_value=110,
        holding_bars=35,
        histogram_fade_bars=3,
        stop_loss_pct=5,
        take_profit_pct=12,
        trailing_stop_pct=8,
        max_holding_bars=80,
    )

    assert reason is None


def test_macd_volume_exit_holds_past_max_bars_while_trend_is_intact():
    reason = get_macd_volume_exit_reason(
        previous_dif=1.4,
        previous_dea=1.0,
        current_dif=1.5,
        current_dea=1.1,
        recent_histogram=[0.3, 0.4, 0.5],
        current_price=125,
        entry_price=100,
        highest_price=126,
        trend_ma_value=110,
        holding_bars=120,
        histogram_fade_bars=3,
        stop_loss_pct=5,
        take_profit_pct=12,
        trailing_stop_pct=8,
        max_holding_bars=80,
    )

    assert reason is None


def test_macd_volume_exit_sells_when_histogram_fades_and_trend_breaks():
    reason = get_macd_volume_exit_reason(
        previous_dif=1.4,
        previous_dea=1.0,
        current_dif=1.2,
        current_dea=1.1,
        recent_histogram=[0.6, 0.4, 0.2],
        current_price=98,
        entry_price=100,
        highest_price=103,
        trend_ma_value=105,
        holding_bars=20,
        histogram_fade_bars=3,
        stop_loss_pct=5,
        take_profit_pct=12,
        trailing_stop_pct=8,
        max_holding_bars=80,
    )

    assert reason == "histogram_fade"


def test_macd_volume_strategy_appears_in_strategy_list():
    main.load_strategy_modules()
    client = TestClient(main.app)

    response = client.get("/strategies")

    assert response.status_code == 200
    strategies = {item["name"]: item for item in response.json()}
    strategy = strategies["macd_volume_divergence_risk_control"]
    assert strategy["display_name"] == "MACD放量背离风控策略"
    assert strategy["parameters"]


def test_is_golden_cross_requires_dif_crossing_up_dea():
    assert is_golden_cross(previous_dif=-0.2, previous_dea=-0.1, current_dif=0.1, current_dea=0.0)
    assert not is_golden_cross(previous_dif=0.2, previous_dea=0.1, current_dif=0.1, current_dea=0.0)
