import pandas as pd
from fastapi.testclient import TestClient

import main
from market_data import DataSourceResult
from strategies.rsi_risk_control import should_enter_long, get_exit_reason


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
        main,
        "fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(
            data=data,
            provider="test",
            warnings=[],
        ),
    )

    def fake_plot(self, filename, open_browser=False):
        with open(filename, "w", encoding="utf-8") as file:
            file.write("<html>plot</html>")

    monkeypatch.setattr(main.Backtest, "plot", fake_plot)

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
