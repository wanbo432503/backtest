from datetime import date

import pandas as pd
import pytest

from market_data import (
    DataSourceResult,
    detect_market,
    fetch_eastmoney_daily_ohlcv,
    fetch_ohlcv,
    fetch_mootdx_ohlcv,
    normalize_symbol,
    parse_eastmoney_kline_payload,
    prepare_ohlcv,
    to_yfinance_symbol,
)
from market_data_cache import load_daily_cache, save_daily_cache


@pytest.fixture(autouse=True)
def isolate_market_data_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path / "market-cache"))
    monkeypatch.setenv("MARKET_DATA_CACHE_ENABLED", "true")
    monkeypatch.setattr(
        "market_data.fetch_mootdx_corporate_actions",
        lambda symbol: pd.DataFrame(),
        raising=False,
    )


def test_detects_a_share_formats():
    assert detect_market("600519") == "CN"
    assert detect_market("SH600519") == "CN"
    assert detect_market("600519.SH") == "CN"
    assert detect_market("000001.SZ") == "CN"


def test_normalizes_a_share_to_plain_code():
    assert normalize_symbol("SH600519").code == "600519"
    assert normalize_symbol("600519.SH").code == "600519"
    assert normalize_symbol("SZ000001").code == "000001"


def test_non_cn_symbols_can_still_be_detected_before_rejection():
    assert normalize_symbol("AAPL").symbol == "AAPL"
    assert normalize_symbol("0700.HK").symbol == "0700.HK"
    assert normalize_symbol("BTC-USD").symbol == "BTC-USD"


def test_fetch_ohlcv_rejects_non_a_share_symbols():
    with pytest.raises(ValueError, match="仅支持 A 股代码"):
        fetch_ohlcv("BTC-USD", "2026-07-01", "2026-07-04", "1d", "auto")


def test_converts_a_share_symbol_for_yfinance_fallback():
    assert to_yfinance_symbol("600519") == "600519.SS"
    assert to_yfinance_symbol("SH600519") == "600519.SS"
    assert to_yfinance_symbol("000001") == "000001.SZ"


def test_parses_eastmoney_kline_payload():
    payload = {
        "rc": 0,
        "data": {
            "klines": [
                "2026-07-03,1205.24,1194.45,1210.14,1185.00,34268,4099266243.00,2.09,-0.71,-8.55,0.27"
            ]
        },
    }

    frame = parse_eastmoney_kline_payload(payload)

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume", "Amount"]
    assert frame.iloc[0]["High"] == 1210.14


def test_auto_a_share_prefers_mootdx(monkeypatch):
    calls = []

    def mootdx_source(*args, **kwargs):
        calls.append("mootdx")
        data = pd.DataFrame(
            {
                "Open": [10],
                "High": [11],
                "Low": [9],
                "Close": [10.5],
                "Volume": [1000],
            },
            index=pd.to_datetime(["2026-07-03"]),
        )
        return DataSourceResult(data=data, provider="mootdx", warnings=["mootdx source"])

    def unused_source(name):
        def source(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} should not be attempted before mootdx")

        return source

    monkeypatch.setattr("market_data.fetch_mootdx_ohlcv", mootdx_source)
    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", unused_source("yfinance"))
    monkeypatch.setattr("market_data.fetch_eastmoney_daily_ohlcv", unused_source("eastmoney"))

    result = fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-04", "1d", "auto")

    assert result.provider == "mootdx"
    assert calls == ["mootdx"]


def test_daily_mootdx_result_is_adjusted_after_raw_cache_is_loaded(monkeypatch):
    raw = pd.DataFrame(
        {
            "Open": [52.5, 35.28],
            "High": [53.8, 35.8],
            "Low": [52.0, 33.0],
            "Close": [53.05, 33.8],
            "Volume": [1_000, 2_000],
        },
        index=pd.to_datetime(["2015-06-18", "2015-06-19"]),
    )
    actions = pd.DataFrame(
        {
            "category": [1],
            "fenhong": [0.8],
            "songzhuangu": [5.0],
            "peigu": [0.0],
            "peigujia": [0.0],
        },
        index=pd.to_datetime(["2015-06-19"]),
    )
    monkeypatch.setattr(
        "market_data.fetch_mootdx_ohlcv",
        lambda *args, **kwargs: DataSourceResult(raw, "mootdx", []),
    )
    monkeypatch.setattr(
        "market_data.fetch_mootdx_corporate_actions",
        lambda symbol: actions,
    )

    result = fetch_ohlcv("SZ002475", "2015-06-18", "2015-06-19", "1d", "mootdx")

    assert result.data.loc["2015-06-18", "RawClose"] == 53.05
    assert result.data.loc["2015-06-18", "Close"] == pytest.approx(35.3133333333)
    assert any("双价格" in warning for warning in result.warnings)
    snapshot = load_daily_cache("sz002475", "mootdx")
    assert snapshot is not None
    assert "RawClose" not in snapshot.data.columns
    assert snapshot.data.loc["2015-06-18", "Close"] == 53.05

    prepared_again = prepare_ohlcv(result.data)
    assert "RawClose" in prepared_again.columns
    assert "AdjFactor" in prepared_again.columns
    assert "CashDividendPer10" in prepared_again.columns


def test_auto_falls_back_when_mootdx_corporate_actions_are_unavailable(monkeypatch):
    calls = []
    raw = pd.DataFrame(
        {
            "Open": [10],
            "High": [11],
            "Low": [9],
            "Close": [10.5],
            "Volume": [1_000],
        },
        index=pd.to_datetime(["2026-07-03"]),
    )

    def mootdx_source(*args, **kwargs):
        calls.append("mootdx")
        return DataSourceResult(raw, "mootdx", [])

    def yfinance_source(*args, **kwargs):
        calls.append("yfinance")
        return DataSourceResult(raw, "yfinance", [])

    monkeypatch.setattr("market_data.fetch_mootdx_ohlcv", mootdx_source)
    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", yfinance_source)
    monkeypatch.setattr(
        "market_data.fetch_mootdx_corporate_actions",
        lambda symbol: (_ for _ in ()).throw(ValueError("xdxr unavailable")),
    )

    result = fetch_ohlcv("SZ002475", "2026-07-03", "2026-07-03", "1d", "auto")

    assert result.provider == "yfinance"
    assert calls == ["mootdx", "yfinance"]
    assert any("xdxr unavailable" in warning for warning in result.warnings)


def test_mootdx_warns_without_rejecting_large_legitimate_move(monkeypatch):
    raw = pd.DataFrame(
        {
            "Open": [52.5, 35.28],
            "High": [53.8, 35.8],
            "Low": [52.0, 33.0],
            "Close": [53.05, 33.8],
            "Volume": [1_000, 2_000],
        },
        index=pd.to_datetime(["2015-06-18", "2015-06-19"]),
    )
    monkeypatch.setattr(
        "market_data.fetch_mootdx_ohlcv",
        lambda *args, **kwargs: DataSourceResult(raw, "mootdx", []),
    )
    monkeypatch.setattr(
        "market_data.fetch_mootdx_corporate_actions",
        lambda symbol: pd.DataFrame(),
    )

    result = fetch_ohlcv("BJ830001", "2015-06-18", "2015-06-19", "1d", "mootdx")

    assert result.provider == "mootdx"
    assert any("大幅波动" in warning for warning in result.warnings)


def test_old_yfinance_cache_without_raw_contract_is_refetched(monkeypatch):
    old = pd.DataFrame(
        {
            "Open": [10],
            "High": [11],
            "Low": [9],
            "Close": [10.5],
            "Volume": [1_000],
        },
        index=pd.to_datetime(["2026-07-03"]),
    )
    save_daily_cache(
        "sz002475",
        "yfinance",
        old,
        [(date(2026, 7, 3), date(2026, 7, 3))],
        [],
    )
    calls = []
    fresh = old.assign(
        **{
            "Adj Close": [10.4],
            "Dividends": [0.0],
            "Stock Splits": [0.0],
        }
    )

    def yfinance_source(*args, **kwargs):
        calls.append("yfinance")
        return DataSourceResult(fresh, "yfinance", [])

    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", yfinance_source)

    result = fetch_ohlcv("SZ002475", "2026-07-03", "2026-07-03", "1d", "yfinance")

    assert calls == ["yfinance"]
    assert result.data.loc["2026-07-03", "RawClose"] == 10.5


def test_eastmoney_requests_raw_prices(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "rc": 0,
                "data": {"klines": ["2026-07-03,10,10.5,11,9,1000,10000"]},
            }

    def fake_get(url, **kwargs):
        captured.update(kwargs["params"])
        return Response()

    monkeypatch.setattr("market_data.requests.get", fake_get)

    fetch_eastmoney_daily_ohlcv(
        normalize_symbol("SZ002475"), "2026-07-03", "2026-07-03"
    )

    assert captured["fqt"] == "0"


def test_auto_a_share_falls_back_to_yfinance_when_mootdx_fails(monkeypatch):
    calls = []

    def broken_mootdx(*args, **kwargs):
        calls.append("mootdx")
        raise ValueError("mootdx unavailable")

    def fallback_yfinance(*args, **kwargs):
        calls.append("yfinance")
        data = pd.DataFrame(
            {
                "Open": [10],
                "High": [11],
                "Low": [9],
                "Close": [10.5],
                "Volume": [1000],
            },
            index=pd.to_datetime(["2026-07-03"]),
        )
        return DataSourceResult(data=data, provider="yfinance", warnings=["fallback source"])

    def unused_eastmoney(*args, **kwargs):
        calls.append("eastmoney")
        raise AssertionError("eastmoney should not be part of the auto fallback chain")

    monkeypatch.setattr("market_data.fetch_mootdx_ohlcv", broken_mootdx)
    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", fallback_yfinance)
    monkeypatch.setattr("market_data.fetch_eastmoney_daily_ohlcv", unused_eastmoney)

    result = fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-04", "1d", "auto")

    assert result.provider == "yfinance"
    assert calls == ["mootdx", "yfinance"]
    assert any("mootdx 获取失败" in warning for warning in result.warnings)


def test_baidu_provider_is_not_supported():
    with pytest.raises(ValueError, match="不支持的数据源或市场组合"):
        fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-04", "1d", "baidu")


def test_mootdx_adapter_handles_duplicate_volume_columns(monkeypatch):
    class DummyQuotes:
        def bars(self, symbol, frequency, start=0, offset=800):
            return pd.DataFrame(
                {
                    "open": [10],
                    "close": [10.5],
                    "high": [11],
                    "low": [9],
                    "vol": [1000],
                    "amount": [12000],
                    "datetime": ["2026-07-03 15:00"],
                    "volume": [1000],
                },
                index=pd.to_datetime(["2026-07-03 15:00"]),
            )

    monkeypatch.setattr("mootdx.quotes.Quotes.factory", lambda market: DummyQuotes())

    result = fetch_mootdx_ohlcv(normalize_symbol("SZ002241"), "1d")

    assert result.provider == "mootdx"
    assert list(result.data.columns) == ["Open", "High", "Low", "Close", "Volume", "Amount"]
    assert result.data.iloc[0]["Volume"] == 1000


def test_mootdx_adapter_paginates_until_requested_start_date(monkeypatch):
    calls = []

    class DummyQuotes:
        def bars(self, symbol, frequency, start, offset):
            calls.append((start, offset))
            if start == 0:
                dates = pd.date_range("2025-09-02 10:30", periods=800, freq="h")
            elif start == 800:
                dates = pd.date_range("2025-07-03 10:30", periods=800, freq="h")
            else:
                return pd.DataFrame()
            return pd.DataFrame(
                {
                    "open": range(len(dates)),
                    "close": range(len(dates)),
                    "high": range(len(dates)),
                    "low": range(len(dates)),
                    "vol": range(len(dates)),
                    "amount": range(len(dates)),
                    "datetime": [date.strftime("%Y-%m-%d %H:%M") for date in dates],
                },
                index=dates,
            )

    monkeypatch.setattr("mootdx.quotes.Quotes.factory", lambda market: DummyQuotes())

    result = fetch_mootdx_ohlcv(
        normalize_symbol("SZ002241"),
        "1h",
        "2025-07-03",
        "2025-10-01",
    )

    assert calls == [(0, 800), (800, 800)]
    assert result.data.index.min() == pd.Timestamp("2025-07-03 10:30")
    assert result.data.index.max() <= pd.Timestamp("2025-10-02")
