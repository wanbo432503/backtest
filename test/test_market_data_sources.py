from market_data import (
    detect_market,
    normalize_symbol,
    parse_baidu_kline_payload,
    parse_eastmoney_kline_payload,
    to_yfinance_symbol,
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


def test_non_cn_symbols_remain_yfinance_compatible():
    assert normalize_symbol("AAPL").symbol == "AAPL"
    assert normalize_symbol("0700.HK").symbol == "0700.HK"
    assert normalize_symbol("BTC-USD").symbol == "BTC-USD"


def test_converts_a_share_symbol_for_yfinance_fallback():
    assert to_yfinance_symbol("600519") == "600519.SS"
    assert to_yfinance_symbol("SH600519") == "600519.SS"
    assert to_yfinance_symbol("000001") == "000001.SZ"


def test_parses_baidu_kline_payload():
    payload = {
        "ResultCode": "0",
        "Result": {
            "newMarketData": {
                "keys": [
                    "timestamp",
                    "time",
                    "open",
                    "close",
                    "volume",
                    "high",
                    "low",
                    "amount",
                ],
                "marketData": "1783353600,2026-07-03,1205.24,1194.45,34268,1210.14,1185.00,4099266243",
            }
        },
    }

    frame = parse_baidu_kline_payload(payload)

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume", "Amount"]
    assert frame.iloc[0]["Close"] == 1194.45


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
