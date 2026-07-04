import stock_search
from stock_search import search_cn_stocks, search_stocks


class DummyTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    @property
    def info(self):
        raise RuntimeError("network disabled in test")


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_search_cn_stocks_uses_free_a_share_name_lookup(monkeypatch):
    monkeypatch.setattr("stock_search.yf.Ticker", DummyTicker)
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
            }
        ],
        raising=False,
    )

    results = search_cn_stocks("中科曙光")

    assert results[0]["symbol"] == "SH603019"
    assert results[0]["name"] == "中科曙光"
    assert results[0]["market"] == "CN"
    assert results[0]["type"] == "remote_name_match"


def test_general_search_includes_remote_a_share_name_lookup(monkeypatch):
    monkeypatch.setattr("stock_search.yf.Ticker", DummyTicker)
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
            }
        ],
        raising=False,
    )

    results = search_stocks("中科曙光")

    assert any(result["symbol"] == "SH603019" for result in results)


def test_general_search_does_not_return_non_a_share_results(monkeypatch):
    monkeypatch.setattr("stock_search.yf.Ticker", DummyTicker)
    monkeypatch.setattr(
        stock_search,
        "_fetch_eastmoney_a_share_suggestions",
        lambda query: [],
        raising=False,
    )

    results = search_stocks("AAPL")

    assert results == []
