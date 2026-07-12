import pandas as pd
from fastapi.testclient import TestClient

import backtest_runner
import main
from market_data import DataSourceResult
from strategy_engine import SimulationPosition, StrategyBarContext
from strategies.rsi_risk_control import (
    RSIConfig,
    STRATEGY_DEFINITION,
    get_exit_reason,
    should_enter_long,
)


def _decision_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0, 100.0, 100.0, 101.0],
            "High": [101.0, 101.0, 101.0, 101.0, 101.0, 102.0],
            "Low": [99.0, 99.0, 99.0, 99.0, 99.0, 100.0],
            "Close": [100.0, 100.0, 100.0, 100.0, 100.0, 101.0],
            "Volume": [1_000_000] * 6,
            "rsi": [50.0, 50.0, 50.0, 50.0, 29.0, 31.0],
            "trend_ma_value": [90.0] * 6,
        },
        index=pd.date_range("2026-01-01", periods=6, freq="D"),
    )


def test_rsi_crossing_buy_threshold_enters_when_trend_allows():
    assert should_enter_long(
        previous_rsi=24,
        current_rsi=31,
        close=12.5,
        trend_ma_value=12,
        cooldown_remaining=0,
        rsi_buy=30,
    )


def test_stop_loss_exit_reason_when_price_breaks_loss_limit():
    reason = get_exit_reason(
        current_price=94,
        entry_price=100,
        previous_rsi=40,
        current_rsi=45,
        close=94,
        trend_ma_value=90,
        holding_bars=3,
        rsi_sell=70,
        stop_loss_pct=5,
        take_profit_pct=12,
        max_holding_bars=120,
    )

    assert reason == "stop_loss"


def test_take_profit_exit_reason_when_price_reaches_profit_target():
    reason = get_exit_reason(
        current_price=113,
        entry_price=100,
        previous_rsi=40,
        current_rsi=45,
        close=113,
        trend_ma_value=90,
        holding_bars=3,
        rsi_sell=70,
        stop_loss_pct=5,
        take_profit_pct=12,
        max_holding_bars=120,
    )

    assert reason == "take_profit"


def test_max_holding_bars_exit_reason_when_position_ages_out():
    reason = get_exit_reason(
        current_price=101,
        entry_price=100,
        previous_rsi=40,
        current_rsi=45,
        close=101,
        trend_ma_value=90,
        holding_bars=120,
        rsi_sell=70,
        stop_loss_pct=5,
        take_profit_pct=12,
        max_holding_bars=120,
    )

    assert reason == "max_holding_bars"


def test_rsi_definition_emits_next_open_entry():
    config = RSIConfig(rsi_period=2, trend_ma=5)
    frame = _decision_frame()
    context = StrategyBarContext(
        symbol="SH603019",
        frame=frame,
        bar_index=5,
        config=config,
    )

    decision = STRATEGY_DEFINITION.evaluate(context)

    assert decision.entry is not None
    assert decision.entry.order_type == "next_open"
    assert decision.entry.suggested_position_pct == 0.95


def test_rsi_definition_emits_exit_and_cooldown_state():
    config = RSIConfig(rsi_period=2, trend_ma=5, stop_loss_pct=5, cooldown_bars=3)
    frame = _decision_frame()
    frame.loc[frame.index[-1], "Close"] = 94.0
    position = SimulationPosition(
        symbol="SH603019",
        shares=100,
        entry_date="2026-01-01",
        entry_price=100.0,
        holding_bars=2,
    )
    context = StrategyBarContext(
        symbol="SH603019",
        frame=frame,
        bar_index=5,
        config=config,
        position=position,
    )

    decision = STRATEGY_DEFINITION.evaluate(context)

    assert decision.exit is not None
    assert decision.exit.reason == "stop_loss"
    assert decision.exit.order_type == "next_open"
    assert decision.next_state == {"cooldown_remaining": 3}


def test_rsi_definition_decrements_cooldown_without_mutating_input_state():
    state = {"cooldown_remaining": 2}
    context = StrategyBarContext(
        symbol="SH603019",
        frame=_decision_frame(),
        bar_index=5,
        config=RSIConfig(rsi_period=2, trend_ma=5),
        state=state,
    )

    decision = STRATEGY_DEFINITION.evaluate(context)

    assert decision.entry is None
    assert decision.next_state == {"cooldown_remaining": 1}
    assert state == {"cooldown_remaining": 2}


def test_rsi_definition_is_prefix_invariant_when_future_rows_change():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    data = pd.DataFrame(
        {
            "Open": [100 + index * 0.1 for index in range(100)],
            "High": [101 + index * 0.1 for index in range(100)],
            "Low": [99 + index * 0.1 for index in range(100)],
            "Close": [100 + index * 0.1 for index in range(100)],
            "Volume": [1_000_000] * 100,
        },
        index=dates,
    )
    config = RSIConfig()
    prefix = STRATEGY_DEFINITION.prepare_frame(data.iloc[:80], config)
    changed = data.copy()
    changed.loc[dates[80]:, "Close"] = 10_000
    full = STRATEGY_DEFINITION.prepare_frame(changed, config).iloc[:80]

    pd.testing.assert_series_equal(prefix["rsi"], full["rsi"])
    pd.testing.assert_series_equal(prefix["trend_ma_value"], full["trend_ma_value"])
    prefix_decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", prefix, 79, config)
    )
    full_decision = STRATEGY_DEFINITION.evaluate(
        StrategyBarContext("SH603019", full, 79, config)
    )
    assert prefix_decision == full_decision


def test_rsi_risk_control_strategy_is_available_via_api(monkeypatch):
    main.load_strategy_modules()
    client = TestClient(main.app)

    dates = pd.date_range("2025-07-03", periods=90, freq="D")
    prices = list(range(90, 120)) + list(range(120, 150)) + list(range(150, 120, -1))
    data = pd.DataFrame(
        {
            "Open": prices,
            "High": [price + 1 for price in prices],
            "Low": [price - 1 for price in prices],
            "Close": prices,
            "Volume": [1000000] * len(prices),
        },
        index=dates,
    )

    monkeypatch.setattr(
        backtest_runner,
        "fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(
            data=data,
            provider="test",
            warnings=[],
        ),
    )

    response = client.post(
        "/backtest",
        json={
            "symbol": "SH603019",
            "start_date": "2025-07-03",
            "end_date": "2025-10-01",
            "strategy_name": "rsi_risk_control",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data_provider"] == "test"
    assert "plot_html" in payload
    assert "交易次数" in payload["stats"]


def test_rsi_risk_control_strategy_appears_in_strategy_list():
    main.load_strategy_modules()
    client = TestClient(main.app)

    response = client.get("/strategies")

    assert response.status_code == 200
    strategies = {item["name"]: item for item in response.json()}
    assert strategies["rsi_risk_control"]["display_name"] == "RSI风控策略"
    assert strategies["rsi_risk_control"]["parameters"]
