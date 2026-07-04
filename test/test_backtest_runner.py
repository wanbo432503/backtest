import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from backtest_runner import run_single_backtest
from market_data import DataSourceResult


def _sample_ohlcv(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2025-07-03", periods=rows, freq="D")
    prices = [100 + index * 0.2 for index in range(rows)]
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [price + 1 for price in prices],
            "Low": [price - 1 for price in prices],
            "Close": prices,
            "Volume": [1000000] * rows,
        },
        index=dates,
    )


def test_run_single_backtest_returns_score(monkeypatch):
    monkeypatch.setattr(
        "backtest_runner.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(_sample_ohlcv(), "test", []),
    )

    def fake_plot(self, filename, open_browser=False):
        with open(filename, "w", encoding="utf-8") as file:
            file.write("<html>plot</html>")

    monkeypatch.setattr("backtest_runner.Backtest.plot", fake_plot)

    result = run_single_backtest(
        symbol="SH603019",
        start_date="2025-07-03",
        end_date="2025-10-01",
        interval="1d",
        strategy_name="sma_cross",
        strategy_registry=main.STRATEGY_REGISTRY,
        initial_cash=10000,
        commission=0.002,
        data_provider="auto",
    )

    assert result.data_provider == "test"
    assert result.plot_html == "<html>plot</html>"
    assert "综合评分" in result.stats
    assert "score" in result.metrics


def test_run_single_backtest_raises_readable_error_for_bad_data(monkeypatch):
    monkeypatch.setattr(
        "backtest_runner.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(
            pd.DataFrame({"Open": [1], "Close": [1]}),
            "test",
            [],
        ),
    )

    with pytest.raises(ValueError, match="数据缺少必要的列"):
        run_single_backtest(
            symbol="SH603019",
            start_date="2025-07-03",
            end_date="2025-10-01",
            interval="1d",
            strategy_name="sma_cross",
            strategy_registry=main.STRATEGY_REGISTRY,
        )


def test_backtest_api_keeps_legacy_response_shape(monkeypatch):
    client = TestClient(main.app)

    monkeypatch.setattr(
        "backtest_runner.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(_sample_ohlcv(), "test", ["source warning"]),
    )

    def fake_plot(self, filename, open_browser=False):
        with open(filename, "w", encoding="utf-8") as file:
            file.write("<html>plot</html>")

    monkeypatch.setattr("backtest_runner.Backtest.plot", fake_plot)

    response = client.post(
        "/backtest",
        json={
            "symbol": "SH603019",
            "start_date": "2025-07-03",
            "end_date": "2025-10-01",
            "strategy_name": "sma_cross",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {
        "plot_html",
        "stats",
        "symbol",
        "interval",
        "data_provider",
        "data_warnings",
    }
    assert payload["data_provider"] == "test"
    assert payload["data_warnings"] == ["source warning"]
    assert "综合评分" in payload["stats"]
