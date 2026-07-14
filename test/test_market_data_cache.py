import pandas as pd

from market_data import DataSourceResult, fetch_ohlcv
from market_data_cache import load_daily_cache, uncovered_date_ranges
from portfolio_data import load_portfolio_ohlcv


def _daily_frame(start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    return pd.DataFrame(
        {
            "Open": [10.0] * len(dates),
            "High": [11.0] * len(dates),
            "Low": [9.0] * len(dates),
            "Close": [10.5] * len(dates),
            "Volume": [1000] * len(dates),
        },
        index=dates,
    )


def test_daily_market_data_cache_serves_repeated_request_without_refetch(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path))
    calls = []

    def source(symbol, start_date, end_date, interval):
        calls.append((start_date, end_date, interval))
        return DataSourceResult(_daily_frame(start_date, end_date), "yfinance", ["fixture"])

    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", source)

    first = fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-05", "1d", "yfinance")
    second = fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-05", "1d", "yfinance")

    assert calls == [("2026-07-01", "2026-07-05", "1d")]
    assert first.cache_status == "miss"
    assert second.cache_hit is True
    assert second.cache_status == "hit"
    assert second.data.equals(first.data)


def test_daily_market_data_cache_fetches_only_uncovered_tail(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path))
    calls = []

    def source(symbol, start_date, end_date, interval):
        calls.append((start_date, end_date))
        return DataSourceResult(_daily_frame(start_date, end_date), "yfinance", [])

    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", source)

    fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-03", "1d", "yfinance")
    result = fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-06", "1d", "yfinance")

    assert calls == [("2026-07-01", "2026-07-03"), ("2026-07-04", "2026-07-06")]
    assert result.cache_status == "extended"
    assert len(result.data) == 6
    snapshot = load_daily_cache("sz002241", "yfinance")
    assert snapshot is not None
    assert snapshot.covers("2026-07-01", "2026-07-06")


def test_portfolio_cache_policy_uses_cached_tail_without_refetch(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path))
    calls = []

    def source(symbol, start_date, end_date, interval):
        calls.append((start_date, end_date))
        return DataSourceResult(_daily_frame(start_date, end_date), "yfinance", [])

    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", source)

    fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-03", "1d", "yfinance")
    result = fetch_ohlcv(
        "SZ002241",
        "2026-07-01",
        "2026-07-04",
        "1d",
        "yfinance",
        prefer_cached_tail=True,
    )

    assert calls == [("2026-07-01", "2026-07-03")]
    assert result.cache_hit is True
    assert result.cache_status == "stale"
    assert result.data.index.max() == pd.Timestamp("2026-07-03")
    assert any(
        "请求截至 2026-07-04" in warning and "行情仅截至 2026-07-03" in warning
        for warning in result.warnings
    )


def test_portfolio_cache_policy_does_not_hide_missing_leading_history(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path))
    calls = []

    def source(symbol, start_date, end_date, interval):
        calls.append((start_date, end_date))
        return DataSourceResult(_daily_frame(start_date, end_date), "yfinance", [])

    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", source)

    fetch_ohlcv("SZ002241", "2026-07-03", "2026-07-05", "1d", "yfinance")
    result = fetch_ohlcv(
        "SZ002241",
        "2026-07-01",
        "2026-07-05",
        "1d",
        "yfinance",
        prefer_cached_tail=True,
    )

    assert calls == [("2026-07-03", "2026-07-05"), ("2026-07-01", "2026-07-02")]
    assert result.cache_status == "extended"
    assert result.data.index.min() == pd.Timestamp("2026-07-01")


def test_portfolio_cache_policy_does_not_use_overly_stale_tail(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path))
    calls = []

    def source(symbol, start_date, end_date, interval):
        calls.append((start_date, end_date))
        return DataSourceResult(_daily_frame(start_date, end_date), "yfinance", [])

    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", source)

    fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-03", "1d", "yfinance")
    result = fetch_ohlcv(
        "SZ002241",
        "2026-07-01",
        "2026-07-31",
        "1d",
        "yfinance",
        prefer_cached_tail=True,
    )

    assert calls == [("2026-07-01", "2026-07-03"), ("2026-07-04", "2026-07-31")]
    assert result.cache_status == "extended"


def test_auto_portfolio_cache_policy_prefers_stale_mootdx_over_yfinance(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path))

    monkeypatch.setattr(
        "market_data.fetch_mootdx_ohlcv",
        lambda symbol, interval, start_date, end_date: DataSourceResult(
            _daily_frame(start_date, end_date),
            "mootdx",
            [],
        ),
    )
    monkeypatch.setattr(
        "market_data.fetch_yfinance_ohlcv",
        lambda symbol, start_date, end_date, interval: DataSourceResult(
            _daily_frame(start_date, end_date),
            "yfinance",
            [],
        ),
    )
    monkeypatch.setattr(
        "market_data.fetch_mootdx_corporate_actions",
        lambda symbol: pd.DataFrame(),
    )

    fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-03", "1d", "mootdx")
    fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-04", "1d", "yfinance")
    result = fetch_ohlcv(
        "SZ002241",
        "2026-07-01",
        "2026-07-04",
        "1d",
        "auto",
        prefer_cached_tail=True,
    )

    assert result.provider == "mootdx"
    assert result.cache_status == "stale"
    assert result.data.index.max() == pd.Timestamp("2026-07-03")


def test_daily_cache_extends_naive_cache_with_timezone_aware_data(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path))
    calls = []

    def source(symbol, start_date, end_date, interval):
        calls.append((start_date, end_date))
        frame = _daily_frame(start_date, end_date)
        if len(calls) > 1:
            frame.index = frame.index.tz_localize("Asia/Shanghai")
        return DataSourceResult(frame, "yfinance", [])

    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", source)

    fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-03", "1d", "yfinance")
    result = fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-06", "1d", "yfinance")

    assert calls == [("2026-07-01", "2026-07-03"), ("2026-07-04", "2026-07-06")]
    assert result.cache_status == "extended"
    assert len(result.data) == 6
    assert result.data.index.tz is None
    assert result.data.index.min() == pd.Timestamp("2026-07-01")
    assert result.data.index.max() == pd.Timestamp("2026-07-06")


def test_intraday_market_data_bypasses_daily_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path))
    calls = []

    def source(symbol, start_date, end_date, interval):
        calls.append(interval)
        return DataSourceResult(_daily_frame(start_date, end_date), "yfinance", [])

    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", source)

    fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-02", "1h", "yfinance")
    fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-02", "1h", "yfinance")

    assert calls == ["1h", "1h"]


def test_portfolio_loader_reuses_cache_written_by_shared_market_entry(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path))
    calls = []

    def source(symbol, start_date, end_date, interval):
        calls.append(symbol)
        return DataSourceResult(_daily_frame(start_date, end_date), "yfinance", [])

    monkeypatch.setattr("market_data.fetch_yfinance_ohlcv", source)

    fetch_ohlcv("SZ002241", "2026-07-01", "2026-07-05", "1d", "yfinance")
    bundle = load_portfolio_ohlcv(
        ["SZ002241"],
        "2026-07-01",
        "2026-07-05",
        provider="yfinance",
    )

    assert calls == ["SZ002241"]
    assert bundle.cache_hits == 1
    assert bundle.cache_misses == 0


def test_uncovered_date_ranges_handles_multiple_cached_segments():
    ranges = [
        (pd.Timestamp("2026-07-02").date(), pd.Timestamp("2026-07-03").date()),
        (pd.Timestamp("2026-07-05").date(), pd.Timestamp("2026-07-06").date()),
    ]

    gaps = uncovered_date_ranges(ranges, "2026-07-01", "2026-07-07")

    assert gaps == [
        (pd.Timestamp("2026-07-01").date(), pd.Timestamp("2026-07-01").date()),
        (pd.Timestamp("2026-07-04").date(), pd.Timestamp("2026-07-04").date()),
        (pd.Timestamp("2026-07-07").date(), pd.Timestamp("2026-07-07").date()),
    ]
