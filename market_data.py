from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import urllib.request

import pandas as pd
import requests
import yfinance as yf


@dataclass(frozen=True)
class NormalizedSymbol:
    raw: str
    symbol: str
    code: str
    market: str
    exchange_prefix: str = ""


@dataclass
class DataSourceResult:
    data: pd.DataFrame
    provider: str
    warnings: list[str]


def detect_market(symbol: str) -> str:
    value = symbol.strip().upper()
    if value.endswith(".HK"):
        return "HK"
    if value.startswith(("SH", "SZ", "BJ")):
        return "CN"
    if value.endswith((".SH", ".SZ", ".BJ")):
        return "CN"
    if value.isdigit() and len(value) == 6:
        return "CN"
    return "US"


def normalize_symbol(symbol: str) -> NormalizedSymbol:
    raw = symbol.strip()
    value = raw.upper()
    market = detect_market(value)
    if market != "CN":
        return NormalizedSymbol(raw=raw, symbol=value, code=value, market=market)

    code = value
    if code.startswith(("SH", "SZ", "BJ")):
        code = code[2:]
    if code.endswith((".SH", ".SZ", ".BJ")):
        code = code.split(".")[0]

    if code.startswith(("6", "9")):
        prefix = "sh"
    elif code.startswith("8"):
        prefix = "bj"
    else:
        prefix = "sz"

    return NormalizedSymbol(raw=raw, symbol=f"{prefix}{code}", code=code, market="CN", exchange_prefix=prefix)


def to_yfinance_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    if normalized.market != "CN":
        return normalized.symbol
    if normalized.exchange_prefix == "sh":
        suffix = "SS"
    elif normalized.exchange_prefix == "bj":
        suffix = "BJ"
    else:
        suffix = "SZ"
    return f"{normalized.code}.{suffix}"


def prepare_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = [str(col).title() for col in frame.columns]

    required_columns = ["Open", "High", "Low", "Close"]
    missing_columns = [col for col in required_columns if col not in frame.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要的列: {missing_columns}")

    if "Volume" not in frame.columns:
        frame["Volume"] = 0

    for col in required_columns + ["Volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "Amount" in frame.columns:
        frame["Amount"] = pd.to_numeric(frame["Amount"], errors="coerce")

    frame = frame.dropna(subset=required_columns)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)

    return frame.sort_index()


def fetch_yfinance_ohlcv(symbol: str, start_date: str, end_date: str, interval: str) -> DataSourceResult:
    yfinance_symbol = to_yfinance_symbol(symbol)
    data = yf.Ticker(yfinance_symbol).history(start=start_date, end=end_date, interval=interval)
    if data.empty:
        raise ValueError("yfinance returned empty data")
    warnings = []
    if yfinance_symbol != symbol:
        warnings.append(f"已将 {symbol} 转换为 yfinance 代码 {yfinance_symbol}。")
    return DataSourceResult(data=prepare_ohlcv(data), provider="yfinance", warnings=warnings)


def parse_baidu_kline_payload(payload: dict) -> pd.DataFrame:
    if str(payload.get("ResultCode", "-1")) != "0":
        raise ValueError(f"百度股市通返回错误: {payload.get('ResultCode')}")

    market_data = payload.get("Result", {}).get("newMarketData", {})
    keys = market_data.get("keys", [])
    raw_rows = market_data.get("marketData", "")
    rows = [row for row in raw_rows.split(";") if row.strip()]

    records = []
    for row in rows:
        values = row.split(",")
        item = dict(zip(keys, values))
        records.append(
            {
                "Date": item.get("time"),
                "Open": item.get("open"),
                "High": item.get("high"),
                "Low": item.get("low"),
                "Close": item.get("close"),
                "Volume": item.get("volume", 0),
                "Amount": item.get("amount", 0),
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("百度股市通返回空 K 线")

    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.set_index("Date")
    return prepare_ohlcv(frame)[["Open", "High", "Low", "Close", "Volume", "Amount"]]


def fetch_baidu_daily_ohlcv(
    symbol: NormalizedSymbol,
    start_date: str,
    end_date: str,
) -> DataSourceResult:
    url = "https://finance.pae.baidu.com/selfselect/getstockquotation"
    params = {
        "all": "1",
        "isIndex": "false",
        "isBk": "false",
        "isBlock": "false",
        "isFutures": "false",
        "isStock": "true",
        "newFormat": "1",
        "group": "quotation_kline_ab",
        "finClientType": "pc",
        "code": symbol.code,
        "start_time": start_date.replace("-", ""),
        "ktype": "1",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.finance-web.v1+json",
        "Origin": "https://gushitong.baidu.com",
        "Referer": "https://gushitong.baidu.com/",
    }
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()

    frame = parse_baidu_kline_payload(response.json())
    frame = frame[(frame.index >= pd.to_datetime(start_date)) & (frame.index <= pd.to_datetime(end_date))]
    if frame.empty:
        raise ValueError("百度股市通在指定时间区间无数据")

    return DataSourceResult(
        data=frame,
        provider="baidu",
        warnings=["百度股市通日 K 线用于 A 股免费回测数据源；复权口径需后续单独校验。"],
    )


def parse_eastmoney_kline_payload(payload: dict) -> pd.DataFrame:
    if payload.get("rc") != 0:
        raise ValueError(f"东方财富 K 线返回错误: {payload.get('rc')}")

    klines = payload.get("data", {}).get("klines", []) or []
    records = []
    for line in klines:
        values = line.split(",")
        records.append(
            {
                "Date": _value_at(values, 0),
                "Open": _value_at(values, 1),
                "Close": _value_at(values, 2),
                "High": _value_at(values, 3),
                "Low": _value_at(values, 4),
                "Volume": _value_at(values, 5),
                "Amount": _value_at(values, 6),
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("东方财富返回空 K 线")

    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.set_index("Date")
    return prepare_ohlcv(frame)[["Open", "High", "Low", "Close", "Volume", "Amount"]]


def fetch_eastmoney_daily_ohlcv(
    symbol: NormalizedSymbol,
    start_date: str,
    end_date: str,
) -> DataSourceResult:
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "secid": _eastmoney_secid(symbol),
        "beg": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
    }
    response = requests.get(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        params=params,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
        timeout=10,
    )
    response.raise_for_status()
    frame = parse_eastmoney_kline_payload(response.json())
    return DataSourceResult(
        data=frame,
        provider="eastmoney",
        warnings=["东方财富前复权日 K 线作为 A 股免费备用数据源；批量请求需控制频率。"],
    )


def fetch_mootdx_ohlcv(symbol: NormalizedSymbol, interval: str) -> DataSourceResult:
    try:
        from mootdx.quotes import Quotes
    except ImportError as exc:
        raise ValueError("mootdx is not installed") from exc

    frequency_map = {
        "1m": 8,
        "5m": 0,
        "15m": 1,
        "30m": 2,
        "60m": 3,
        "1h": 3,
        "1d": 9,
        "1wk": 5,
        "1mo": 6,
    }
    frequency = frequency_map.get(interval)
    if frequency is None:
        raise ValueError(f"mootdx 不支持该频率: {interval}")

    client = Quotes.factory(market="std")
    data = client.bars(symbol=symbol.code, frequency=frequency, offset=800)
    if data is None or data.empty:
        raise ValueError("mootdx returned empty data")

    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "vol": "Volume",
            "amount": "Amount",
        }
    )
    if "datetime" in data.columns:
        data = data.set_index(pd.to_datetime(data["datetime"]))

    return DataSourceResult(
        data=prepare_ohlcv(data),
        provider="mootdx",
        warnings=["mootdx bars 返回不复权原始价，跨除权除息日回测需谨慎。"],
    )


def fetch_tencent_quote(codes: list[str]) -> dict[str, dict]:
    prefixed = []
    for raw_code in codes:
        normalized = normalize_symbol(raw_code)
        if normalized.market == "CN":
            prefixed.append(f"{normalized.exchange_prefix}{normalized.code}")

    if not prefixed:
        return {}

    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(request, timeout=10).read().decode("gbk", errors="replace")

    result = {}
    for line in data.strip().split(";"):
        if not line.strip() or "=" not in line or '"' not in line:
            continue
        key = line.split("=")[0].split("_")[-1]
        values = line.split('"')[1].split("~")
        if len(values) < 53:
            continue
        code = key[2:]
        result[code] = {
            "name": values[1],
            "price": _float_or_zero(values[3]),
            "last_close": _float_or_zero(values[4]),
            "open": _float_or_zero(values[5]),
            "change_amt": _float_or_zero(values[31]),
            "change_pct": _float_or_zero(values[32]),
            "high": _float_or_zero(values[33]),
            "low": _float_or_zero(values[34]),
            "amount_wan": _float_or_zero(values[37]),
            "turnover_pct": _float_or_zero(values[38]),
            "pe_ttm": _float_or_zero(values[39]),
            "amplitude_pct": _float_or_zero(values[43]),
            "mcap_yi": _float_or_zero(values[44]),
            "float_mcap_yi": _float_or_zero(values[45]),
            "pb": _float_or_zero(values[46]),
            "limit_up": _float_or_zero(values[47]),
            "limit_down": _float_or_zero(values[48]),
            "vol_ratio": _float_or_zero(values[49]),
            "pe_static": _float_or_zero(values[52]),
        }
    return result


def _value_at(values: list[str], index: int) -> str:
    return values[index] if len(values) > index else ""


def _float_or_zero(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _eastmoney_secid(symbol: NormalizedSymbol) -> str:
    prefix = "1" if symbol.exchange_prefix == "sh" else "0"
    return f"{prefix}.{symbol.code}"


def fetch_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    provider: str = "auto",
) -> DataSourceResult:
    normalized = normalize_symbol(symbol)
    normalized_provider = provider.lower()
    attempts: list[Callable[[], DataSourceResult]] = []

    if normalized_provider in ("auto", "baidu") and normalized.market == "CN" and interval == "1d":
        attempts.append(lambda: fetch_baidu_daily_ohlcv(normalized, start_date, end_date))
    if normalized_provider in ("auto", "eastmoney") and normalized.market == "CN" and interval == "1d":
        attempts.append(lambda: fetch_eastmoney_daily_ohlcv(normalized, start_date, end_date))
    if normalized_provider in ("auto", "mootdx") and normalized.market == "CN":
        attempts.append(lambda: fetch_mootdx_ohlcv(normalized, interval))
    if normalized_provider in ("auto", "yfinance"):
        attempts.append(lambda: fetch_yfinance_ohlcv(symbol, start_date, end_date, interval))

    errors = []
    for attempt in attempts:
        try:
            result = attempt()
            if not result.data.empty:
                return result
        except Exception as exc:
            errors.append(str(exc))

    if not attempts:
        raise ValueError(f"不支持的数据源或市场组合: provider={provider}, symbol={symbol}, interval={interval}")
    raise ValueError("所有数据源均获取失败: " + " | ".join(errors))
