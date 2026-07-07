import pandas as pd
import pytest

from market_data import DataSourceResult
from test.fixtures.portfolio_ohlcv import build_ohlcv_frame, build_portfolio_ohlcv_fixture
from portfolio_data import load_portfolio_ohlcv


def test_load_portfolio_ohlcv_fetches_each_symbol_and_preserves_warnings(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])
    calls = []

    def fake_fetch(symbol, start_date, end_date, interval, provider):
        calls.append((symbol, start_date, end_date, interval, provider))
        return DataSourceResult(
            data=fixture[symbol],
            provider="mootdx",
            warnings=[f"{symbol} source warning"],
        )

    monkeypatch.setattr("portfolio_data.fetch_ohlcv", fake_fetch)

    bundle = load_portfolio_ohlcv(
        ["SH603019", "SZ002241"],
        "2025-01-01",
        "2025-12-31",
        provider="auto",
        min_history_bars=120,
    )

    assert calls == [
        ("SH603019", "2025-01-01", "2025-12-31", "1d", "auto"),
        ("SZ002241", "2025-01-01", "2025-12-31", "1d", "auto"),
    ]
    assert set(bundle.data_by_symbol) == {"SH603019", "SZ002241"}
    assert bundle.providers == {"SH603019": "mootdx", "SZ002241": "mootdx"}
    assert "SH603019 source warning" in bundle.warnings
    assert "SZ002241 source warning" in bundle.warnings


def test_load_portfolio_ohlcv_keeps_successful_symbols_when_one_fetch_fails(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019"])

    def fake_fetch(symbol, *args):
        if symbol == "SZ002241":
            raise ValueError("temporary unavailable")
        return DataSourceResult(data=fixture[symbol], provider="yfinance", warnings=[])

    monkeypatch.setattr("portfolio_data.fetch_ohlcv", fake_fetch)

    bundle = load_portfolio_ohlcv(["SH603019", "SZ002241"], "2025-01-01", "2025-12-31")

    assert list(bundle.data_by_symbol) == ["SH603019"]
    assert any("SZ002241 获取失败: temporary unavailable" in warning for warning in bundle.warnings)


def test_load_portfolio_ohlcv_raises_when_all_symbols_fail(monkeypatch):
    def fake_fetch(symbol, *args):
        raise ValueError(f"{symbol} unavailable")

    monkeypatch.setattr("portfolio_data.fetch_ohlcv", fake_fetch)

    with pytest.raises(ValueError, match="所有组合标的数据源均获取失败"):
        load_portfolio_ohlcv(["SH603019", "SZ002241"], "2025-01-01", "2025-12-31")


def test_load_portfolio_ohlcv_drops_insufficient_history_with_warning(monkeypatch):
    short_frame = build_ohlcv_frame(periods=20)

    def fake_fetch(symbol, *args):
        return DataSourceResult(data=short_frame, provider="mootdx", warnings=[])

    monkeypatch.setattr("portfolio_data.fetch_ohlcv", fake_fetch)

    with pytest.raises(ValueError, match="所有组合标的数据源均获取失败"):
        load_portfolio_ohlcv(["SH603019"], "2025-01-01", "2025-12-31", min_history_bars=120)


def test_load_portfolio_ohlcv_prepares_columns_for_runner(monkeypatch):
    raw_frame = pd.DataFrame(
        {
            "open": [10, 11],
            "high": [11, 12],
            "low": [9, 10],
            "close": [10.5, 11.5],
        },
        index=["2025-01-02", "2025-01-03"],
    )

    def fake_fetch(symbol, *args):
        return DataSourceResult(data=raw_frame, provider="mootdx", warnings=[])

    monkeypatch.setattr("portfolio_data.fetch_ohlcv", fake_fetch)

    bundle = load_portfolio_ohlcv(["SH603019"], "2025-01-01", "2025-12-31", min_history_bars=1)

    data = bundle.data_by_symbol["SH603019"]
    assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data["Volume"].tolist() == [0, 0]


def test_load_portfolio_ohlcv_normalizes_timezone_aware_index(monkeypatch):
    raw_frame = pd.DataFrame(
        {
            "Open": [10, 11],
            "High": [11, 12],
            "Low": [9, 10],
            "Close": [10.5, 11.5],
            "Volume": [1000, 1200],
        },
        index=pd.date_range("2025-01-02", periods=2, freq="D", tz="Asia/Shanghai"),
    )

    def fake_fetch(symbol, *args):
        return DataSourceResult(data=raw_frame, provider="yfinance", warnings=[])

    monkeypatch.setattr("portfolio_data.fetch_ohlcv", fake_fetch)

    bundle = load_portfolio_ohlcv(["SZ002241"], "2025-01-01", "2025-12-31", min_history_bars=1)

    data = bundle.data_by_symbol["SZ002241"]
    assert data.index.tz is None
    assert data.index.tolist() == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]


def test_load_portfolio_ohlcv_rejects_non_daily_interval():
    with pytest.raises(ValueError, match="组合回测 MVP 仅支持日线"):
        load_portfolio_ohlcv(["SH603019"], "2025-01-01", "2025-12-31", interval="1h")


def test_load_portfolio_ohlcv_applies_batch_rate_limits_and_reports_progress(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH600000", "SH603019", "SZ000001"])
    fetch_calls = []
    sleep_calls = []
    progress_events = []

    def fake_fetch(symbol, start_date, end_date, interval, provider):
        fetch_calls.append(symbol)
        return DataSourceResult(data=fixture[symbol], provider="fixture", warnings=[])

    monkeypatch.setattr("portfolio_data.fetch_ohlcv", fake_fetch)

    bundle = load_portfolio_ohlcv(
        ["SH600000", "SH603019", "SZ000001"],
        "2025-01-01",
        "2025-12-31",
        batch_size=2,
        batch_delay_seconds=0.5,
        request_delay_seconds=0.1,
        sleeper=sleep_calls.append,
        progress_callback=progress_events.append,
    )

    assert fetch_calls == ["SH600000", "SH603019", "SZ000001"]
    assert sleep_calls == [0.1, 0.5]
    assert set(bundle.data_by_symbol) == {"SH600000", "SH603019", "SZ000001"}
    assert progress_events[0]["phase"] == "loading_ohlcv"
    assert progress_events[-1] == {
        "phase": "loading_ohlcv",
        "total_symbols": 3,
        "loaded_count": 3,
        "failed_count": 0,
        "current_symbol": "SZ000001",
    }
