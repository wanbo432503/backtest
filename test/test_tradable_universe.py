import stock_search
from stock_search import search_stocks
from tradable_universe import (
    TradableUniversePolicy,
    normalize_tradable_symbol,
    validate_tradable_symbol,
    validate_universe,
)


def test_normalize_tradable_symbol_accepts_common_60_and_00_forms():
    assert normalize_tradable_symbol("600000") == "SH600000"
    assert normalize_tradable_symbol("SH600000") == "SH600000"
    assert normalize_tradable_symbol("600000.SH") == "SH600000"
    assert normalize_tradable_symbol("000001") == "SZ000001"
    assert normalize_tradable_symbol("SZ000001") == "SZ000001"
    assert normalize_tradable_symbol("000001.SZ") == "SZ000001"


def test_validate_tradable_symbol_allows_only_60_and_00_stock_codes():
    allowed = ["SH600000", "SH601318", "SH603019", "SH605001", "SZ000001", "SZ001696", "SZ002241"]

    for symbol in allowed:
        result = validate_tradable_symbol(symbol)
        assert result.ok, symbol
        assert result.normalized_symbol == symbol
        assert result.reason is None


def test_validate_tradable_symbol_rejects_unsupported_boards_and_funds():
    rejected = {
        "SZ300750": "unsupported_board",
        "SZ301269": "unsupported_board",
        "SH688001": "unsupported_board",
        "SH689009": "unsupported_board",
        "BJ430047": "unsupported_board",
        "430047": "unsupported_board",
        "830799": "unsupported_board",
        "920001": "unsupported_board",
        "SH510300": "fund_or_etf",
        "SH511880": "fund_or_etf",
        "SH512000": "fund_or_etf",
        "SZ159915": "fund_or_etf",
        "SZ160119": "fund_or_etf",
        "SH000001": "unsupported_board",
        "SH000002": "unsupported_board",
        "SZ600000": "unsupported_board",
        "AAPL": "not_a_share",
    }

    for symbol, reason in rejected.items():
        result = validate_tradable_symbol(symbol)
        assert not result.ok, symbol
        assert result.reason == reason


def test_validate_universe_dedupes_symbols_and_reports_duplicates():
    result = validate_universe(["SH603019", "603019", "SZ002241"])

    assert result.ok
    assert result.accepted_symbols == ["SH603019", "SZ002241"]
    assert [row.reason for row in result.rejected] == ["duplicate_symbol"]


def test_validate_universe_rejects_more_than_four_unique_symbols():
    result = validate_universe(["SH600000", "SH601318", "SH603019", "SH605001", "SZ000001"])

    assert not result.ok
    assert any(row.reason == "too_many_symbols" for row in result.rejected)


def test_validate_universe_can_use_custom_policy():
    policy = TradableUniversePolicy(max_symbols=2)

    result = validate_universe(["SH603019", "SZ002241", "SH600000"], policy=policy)

    assert not result.ok
    assert any(row.reason == "too_many_symbols" for row in result.rejected)


def test_portfolio_mode_search_marks_tradable_results(monkeypatch):
    monkeypatch.setattr("stock_search.yf.Ticker", lambda symbol: None)
    monkeypatch.setattr(
        stock_search,
        "_fetch_eastmoney_a_share_suggestions",
        lambda query: [
            {
                "symbol": "SH603019",
                "name": "中科曙光",
                "market": "CN",
                "sector": "N/A",
                "country": "China",
                "type": "remote_name_match",
            },
            {
                "symbol": "SZ300750",
                "name": "宁德时代",
                "market": "CN",
                "sector": "N/A",
                "country": "China",
                "type": "remote_name_match",
            },
        ],
        raising=False,
    )

    results = search_stocks("测试", market="CN", portfolio_mode=True)
    by_symbol = {row["symbol"]: row for row in results}

    assert by_symbol["SH603019"]["tradable"] is True
    assert by_symbol["SH603019"]["tradable_reason"] is None
    assert by_symbol["SZ300750"]["tradable"] is False
    assert by_symbol["SZ300750"]["tradable_reason"] == "unsupported_board"
