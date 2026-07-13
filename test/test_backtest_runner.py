from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
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
    assert "Volume" in result.plot_html
    assert "综合评分" in result.stats
    assert "score" in result.metrics
    assert result.metrics["score"] == calculate_score(
        result.metrics["annual_return_pct"],
        result.metrics["sharpe"],
        result.metrics["max_drawdown_pct"],
    )


def test_single_backtest_reports_benchmark_and_simulation_details(monkeypatch):
    monkeypatch.setattr(
        "backtest_runner.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(_sample_ohlcv(), "test", []),
    )

    result = run_single_backtest(
        symbol="SH603019",
        start_date="2025-07-03",
        end_date="2025-10-01",
        strategy_name="ma_trend_risk_control",
        strategy_library=get_strategy_library(),
    )
    payload = result.to_api_response()

    assert result.stats["基准收益率"] == "17.80%"
    assert payload["summary"]["benchmark_return_pct"] == pytest.approx(17.8)
    assert payload["equity_curve"]
    assert isinstance(payload["positions"], list)
    assert isinstance(payload["trades"], list)
    assert isinstance(payload["signal_events"], list)


def test_single_stats_use_full_period_exposure_not_final_position():
    stats = backtest_runner._format_stats(
        {
            "total_return_pct": 10,
            "max_drawdown_pct": 5,
            "benchmark_return_pct": 12.5,
            "exposure_time_pct": 37.5,
            "final_gross_exposure": 0,
        },
        {"score": 1},
    )

    assert stats["基准收益率"] == "12.50%"
    assert stats["持仓时间"] == "37.50%"


def test_plot_html_uses_native_backtesting_sections():
    data = _sample_ohlcv(40)
    equity_curve = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "equity": 10_000 + index * 20 - max(0, index - 25) * 30,
        }
        for index, date in enumerate(data.index)
    ]
    trades = [
        {
            "date": data.index[5].strftime("%Y-%m-%d"),
            "side": "buy",
            "shares": 50,
            "price": float(data.iloc[5]["Open"]),
            "cost": 5,
            "reason": "signal",
            "pnl": None,
        },
        {
            "date": data.index[30].strftime("%Y-%m-%d"),
            "side": "sell",
            "shares": 50,
            "price": float(data.iloc[30]["Open"]),
            "cost": 5,
            "reason": "signal_exit",
            "pnl": 240,
        },
    ]

    html = backtest_runner._render_backtesting_plot_html(
        data,
        equity_curve,
        trades,
        initial_cash=10_000,
    )

    assert "Equity" in html
    assert "Profit / Loss" in html
    assert "Trades (1)" in html
    assert "Volume" in html


def test_plot_html_calls_backtesting_plot(monkeypatch):
    captured = {}

    def fake_plot(self, **kwargs):
        captured.update(kwargs)
        Path(kwargs["filename"]).write_text("<html>native backtesting plot</html>")

    monkeypatch.setattr(backtest_runner.Backtest, "plot", fake_plot)

    html = backtest_runner._render_backtesting_plot_html(
        _sample_ohlcv(),
        [{"date": "2025-07-03", "equity": 10_000}],
        [],
        initial_cash=10_000,
    )

    assert html == "<html>native backtesting plot</html>"
    assert captured["open_browser"] is False
    assert captured["plot_equity"] is True
    assert captured["plot_pl"] is True
    assert captured["plot_volume"] is True
    assert captured["plot_trades"] is True


def test_backtesting_plot_maps_raw_trade_prices_to_adjusted_candles():
    data = _sample_ohlcv(4)
    data["RawOpen"] = data["Open"] / 2
    data["RawHigh"] = data["High"] / 2
    data["RawLow"] = data["Low"] / 2
    data["RawClose"] = data["Close"] / 2
    data["AdjFactor"] = 2.0
    trades = [
        {
            "date": data.index[1].strftime("%Y-%m-%d"),
            "side": "buy",
            "shares": 10,
            "price": float(data.iloc[1]["RawOpen"]),
            "cost": 0,
            "pnl": None,
        },
        {
            "date": data.index[2].strftime("%Y-%m-%d"),
            "side": "sell",
            "shares": 15,
            "price": float(data.iloc[2]["RawOpen"]),
            "cost": 0,
            "pnl": 1,
        },
    ]

    plot_trades = backtest_runner._backtesting_trade_frame(data, trades)

    assert plot_trades.iloc[0]["EntryPrice"] == data.iloc[1]["Open"]
    assert plot_trades.iloc[0]["ExitPrice"] == data.iloc[2]["Open"]
    assert plot_trades.iloc[0]["Size"] == 15
    assert plot_trades.iloc[0]["ReturnPct"] == pytest.approx(
        1 / (10 * float(data.iloc[1]["RawOpen"]))
    )


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
    assert "Volume" in result.plot_html
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
        "summary",
        "equity_curve",
        "positions",
        "trades",
        "signal_events",
        "diagnostics",
    }
    assert payload["data_provider"] == "test"
    assert payload["data_warnings"] == ["source warning"]
    assert "corporate_action_events" in payload["diagnostics"]
    assert "综合评分" in payload["stats"]
    assert "Volume" in payload["plot_html"]


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
