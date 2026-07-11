from pathlib import Path

from stock_universe_provider import (
    StockUniverseRecord,
    _builtin_universe_records,
    fetch_akshare_universe,
    fetch_remote_universe,
    filter_tradable_universe_records,
    get_default_stock_universe,
    load_cached_universe,
    save_cached_universe,
)


def test_filter_tradable_universe_records_keeps_only_60_00_ordinary_stocks():
    records = [
        StockUniverseRecord(symbol="SH600000", name="浦发银行", exchange="SH", code_prefix="60", source="test"),
        StockUniverseRecord(symbol="SZ000001", name="平安银行", exchange="SZ", code_prefix="00", source="test"),
        StockUniverseRecord(symbol="SZ002241", name="歌尔股份", exchange="SZ", code_prefix="00", source="test"),
        StockUniverseRecord(symbol="SZ300750", name="宁德时代", exchange="SZ", code_prefix="30", source="test"),
        StockUniverseRecord(symbol="SH688001", name="华兴源创", exchange="SH", code_prefix="68", source="test"),
        StockUniverseRecord(symbol="BJ430047", name="北交样本", exchange="BJ", code_prefix="43", source="test"),
        StockUniverseRecord(symbol="SH510300", name="沪深300ETF", exchange="SH", code_prefix="51", source="test"),
        StockUniverseRecord(symbol="SH000001", name="上证指数", exchange="SH", code_prefix="00", source="test"),
    ]

    filtered = filter_tradable_universe_records(records)

    assert [record.symbol for record in filtered] == ["SH600000", "SZ000001", "SZ002241"]


def test_universe_cache_round_trips_symbol_metadata(tmp_path: Path):
    cache_path = tmp_path / "universe.json"
    records = [
        StockUniverseRecord(
            symbol="SH600000",
            name="浦发银行",
            exchange="SH",
            code_prefix="60",
            status="active",
            source="fixture",
            refreshed_at="2026-07-07T00:00:00",
        )
    ]

    save_cached_universe(records, cache_path)
    loaded = load_cached_universe(cache_path)

    assert loaded == records


def test_get_default_stock_universe_uses_cache_before_fetching(tmp_path: Path):
    cache_path = tmp_path / "universe.json"
    save_cached_universe(
        [
            StockUniverseRecord(
                symbol="SZ000001",
                name="平安银行",
                exchange="SZ",
                code_prefix="00",
                source="cache",
                refreshed_at="2026-07-07T00:00:00",
            )
        ],
        cache_path,
    )

    universe = get_default_stock_universe(
        cache_path=cache_path,
        fetcher=lambda: (_ for _ in ()).throw(AssertionError("fetcher should not run")),
    )

    assert [record.symbol for record in universe.records] == ["SZ000001"]
    assert universe.source == "cache"


def test_get_default_stock_universe_falls_back_to_builtin_records(tmp_path: Path):
    universe = get_default_stock_universe(
        cache_path=tmp_path / "missing.json",
        refresh=True,
        fetcher=lambda: (_ for _ in ()).throw(RuntimeError("network down")),
    )

    assert universe.records
    assert all(record.symbol.startswith(("SH60", "SZ00")) for record in universe.records)
    assert universe.source == "builtin"
    assert any("network down" in warning for warning in universe.warnings)


def test_builtin_universe_uses_real_a_share_symbols_instead_of_proxies():
    records = filter_tradable_universe_records(_builtin_universe_records())
    symbols = {record.symbol for record in records}

    assert "SH601318" in symbols
    assert "SZ002594" in symbols
    assert "SZ002502" not in symbols
    assert "SZ001898" not in symbols


def test_builtin_cache_is_refreshed_instead_of_reused_forever(tmp_path: Path):
    cache_path = tmp_path / "universe.json"
    save_cached_universe(
        [StockUniverseRecord(symbol="SZ002502", name="旧代理", source="builtin")],
        cache_path,
    )

    universe = get_default_stock_universe(
        cache_path=cache_path,
        fetcher=lambda: [
            StockUniverseRecord(symbol="SH600000", name="浦发银行", source="remote-test")
        ],
    )

    assert [record.symbol for record in universe.records] == ["SH600000"]
    assert universe.source == "remote-test"
    assert "builtin_cache_refresh_required" in universe.warnings


def test_fetch_akshare_universe_normalizes_code_and_filters_board(monkeypatch):
    import akshare
    import pandas as pd

    monkeypatch.setattr(
        akshare,
        "stock_info_a_code_name",
        lambda: pd.DataFrame(
            [
                {"code": "600000", "name": "浦发银行"},
                {"code": "1", "name": "平安银行"},
                {"code": "300750", "name": "宁德时代"},
            ]
        ),
    )

    records = fetch_akshare_universe()

    assert [record.symbol for record in records] == ["SH600000", "SZ000001"]
    assert all(record.source == "akshare" for record in records)


def test_remote_universe_falls_through_from_mootdx_to_akshare(monkeypatch):
    monkeypatch.setattr(
        "stock_universe_provider.fetch_mootdx_universe",
        lambda: (_ for _ in ()).throw(RuntimeError("mootdx unavailable")),
    )
    monkeypatch.setattr(
        "stock_universe_provider.fetch_akshare_universe",
        lambda: [StockUniverseRecord(symbol="SZ002241", name="歌尔股份", source="akshare")],
    )

    records = fetch_remote_universe()

    assert [record.symbol for record in records] == ["SZ002241"]
