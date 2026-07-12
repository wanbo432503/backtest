import pandas as pd
import pytest
from dataclasses import replace
from fastapi.testclient import TestClient

import main
import backtest_runner
from analytics import calculate_score
from backtest_runner import BacktestResult, run_single_backtest
from market_data import DataSourceResult
from strategy_library import get_strategy_library
from strategy_library import StrategyLibrary
from strategies.volume_divergence_rsi_long import STRATEGY_DEFINITION as LONG_DEFINITION


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


def _prepared_long_trade_frame(data, config):
    frame = data.copy()
    frame["ma_value"] = 100.0
    frame["average_volume"] = 1_000_000.0
    frame["volume_confirmed"] = False
    frame["macd_dif"] = -1.0
    frame["bottom_divergence"] = False
    frame["bottom_divergence_recent"] = False
    frame["rsi"] = 50.0
    previous, signal, fill, stop = frame.index[59:63]
    frame.loc[previous, ["Close", "ma_value", "rsi"]] = [99, 100, 29]
    frame.loc[signal, ["Open", "High", "Low", "Close", "ma_value", "Volume", "volume_confirmed", "bottom_divergence_recent", "rsi"]] = [
        100, 102, 99, 101, 100, 1_300_000, True, True, 31,
    ]
    frame.loc[fill, ["Open", "High", "Low", "Close"]] = [99, 106, 98.5, 105]
    frame.loc[stop, ["Open", "High", "Low", "Close"]] = [104, 105, 103, 104]
    return frame


def test_run_single_backtest_returns_score(monkeypatch):
    monkeypatch.setattr(
        "backtest_runner.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(_sample_ohlcv(), "test", []),
    )

    result = run_single_backtest(
        symbol="SH603019",
        start_date="2025-07-03",
        end_date="2025-10-01",
        interval="1d",
        strategy_name="ma_trend_risk_control",
        strategy_library=get_strategy_library(),
        initial_cash=10000,
        commission=0.002,
        data_provider="auto",
    )

    assert result.data_provider == "test"
    assert "<html" in result.plot_html.lower()
    assert "Unified Strategy Backtest" in result.plot_html
    assert "综合评分" in result.stats
    assert "score" in result.metrics
    assert result.metrics["score"] == calculate_score(
        result.metrics["annual_return_pct"],
        result.metrics["sharpe"],
        result.metrics["max_drawdown_pct"],
    )


def test_plot_root_layout_stretches_to_iframe_width(monkeypatch):
    captured = {}

    def fake_file_html(model, resources, title):
        captured["model"] = model
        return "<html>plot</html>"

    monkeypatch.setattr(backtest_runner, "file_html", fake_file_html)

    backtest_runner._render_plot_html(
        _sample_ohlcv(),
        [{"date": "2025-07-03", "equity": 10_000}],
        [],
    )

    assert captured["model"].sizing_mode == "stretch_width"


def test_new_volume_divergence_rsi_strategy_runs_in_single_stock_mode(monkeypatch):
    monkeypatch.setattr(
        "backtest_runner.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(_sample_ohlcv(), "test", []),
    )

    result = run_single_backtest(
        symbol="SH603019",
        start_date="2025-07-03",
        end_date="2025-10-01",
        strategy_name="volume_divergence_rsi_long",
        strategy_library=get_strategy_library(),
    )

    assert result.symbol == "SH603019"
    assert "Unified Strategy Backtest" in result.plot_html
    assert "score" in result.metrics


def test_new_strategy_single_stock_path_fills_and_applies_trailing_stop(monkeypatch):
    monkeypatch.setattr(
        "backtest_runner.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(_sample_ohlcv(), "test", []),
    )
    definition = replace(
        LONG_DEFINITION,
        prepare_frame=_prepared_long_trade_frame,
    )

    result = run_single_backtest(
        symbol="SH603019",
        start_date="2025-07-03",
        end_date="2025-10-01",
        strategy_name=definition.strategy_id,
        strategy_library=StrategyLibrary([definition]),
        initial_cash=100_000,
    )

    assert result.metrics["trades"] == 1
    assert result.stats["交易次数"] == 1


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
            strategy_name="ma_trend_risk_control",
            strategy_library=get_strategy_library(),
        )


def test_run_single_backtest_validates_strategy_parameters_before_simulation(monkeypatch):
    monkeypatch.setattr(
        "backtest_runner.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(_sample_ohlcv(), "test", []),
    )

    with pytest.raises(ValueError, match="extra_forbidden"):
        run_single_backtest(
            symbol="SH603019",
            start_date="2025-07-03",
            end_date="2025-10-01",
            strategy_name="rsi_risk_control",
            strategy_library=get_strategy_library(),
            strategy_params={"unknown_parameter": 1},
        )


def test_backtest_api_keeps_legacy_response_shape(monkeypatch):
    client = TestClient(main.app)

    monkeypatch.setattr(
        "backtest_runner.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(_sample_ohlcv(), "test", ["source warning"]),
    )

    response = client.post(
        "/backtest",
        json={
            "symbol": "SH603019",
            "start_date": "2025-07-03",
            "end_date": "2025-10-01",
            "strategy_name": "ma_trend_risk_control",
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
    assert "Unified Strategy Backtest" in payload["plot_html"]


def test_backtest_api_passes_strategy_params_to_runner(monkeypatch):
    captured = {}

    def fake_run_single_backtest(**kwargs):
        captured.update(kwargs)
        return BacktestResult(
            plot_html="<html>plot</html>",
            stats={"综合评分": "1.00"},
            metrics={"score": 1},
            symbol=kwargs["symbol"],
            interval=kwargs["interval"],
            data_provider="test",
            data_warnings=[],
        )

    monkeypatch.setattr(main, "run_single_backtest", fake_run_single_backtest)
    client = TestClient(main.app)

    response = client.post(
        "/backtest",
        json={
            "symbol": "SH603019",
            "start_date": "2025-07-03",
            "end_date": "2025-10-01",
            "strategy_name": "rsi_risk_control",
            "strategy_params": {"rsi_period": 6, "stop_loss_pct": 3},
            "risk_config": {"position_pct": 0.95},
            "a_share_config": {"lot_size": 100},
        },
    )

    assert response.status_code == 200
    assert captured["strategy_params"] == {"rsi_period": 6, "stop_loss_pct": 3}
    assert captured["strategy_library"] is main.STRATEGY_LIBRARY
