from pathlib import Path

from stock_universe_provider import (
    StockUniverseRecord,
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
