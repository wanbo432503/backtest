from fastapi.testclient import TestClient

import main
from strategies.ma_trend_risk_control import get_ma_exit_reason, should_enter_ma_trend


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
