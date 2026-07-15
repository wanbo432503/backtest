from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
import urllib.request

import pandas as pd
import requests
import yfinance as yf

from market_data_cache import (
    daily_cache_enabled,
    load_daily_cache,
    save_daily_cache,
    slice_daily_cache,
    uncovered_date_ranges,
)
from price_adjustment import apply_corporate_actions, detect_unexplained_discontinuities

YFINANCE_RAW_CACHE_MARKER = "internal:yfinance_raw_actions_v1"
PORTFOLIO_CACHE_MAX_STALE_DAYS = 14


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
    cache_hit: bool = False
    cache_status: str = "disabled"
    corporate_action_cache_status: str = "disabled"


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
    canonical_columns = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "amount": "Amount",
        "rawopen": "RawOpen",
        "rawhigh": "RawHigh",
        "rawlow": "RawLow",
        "rawclose": "RawClose",
        "adjfactor": "AdjFactor",
        "cashdividendper10": "CashDividendPer10",
        "bonussharesper10": "BonusSharesPer10",
        "rightssharesper10": "RightsSharesPer10",
        "rightsprice": "RightsPrice",
        "dividends": "Dividends",
        "stock splits": "Stock Splits",
        "adj close": "Adj Close",
    }
    frame.columns = [
        canonical_columns.get(str(column).strip().lower(), str(column).title())
        for column in frame.columns
    ]

    required_columns = ["Open", "High", "Low", "Close"]
    missing_columns = [col for col in required_columns if col not in frame.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要的列: {missing_columns}")

    if "Volume" not in frame.columns:
        frame["Volume"] = 0

    numeric_columns = [
        *required_columns,
        "Volume",
        "RawOpen",
        "RawHigh",
        "RawLow",
        "RawClose",
        "AdjFactor",
        "CashDividendPer10",
        "BonusSharesPer10",
        "RightsSharesPer10",
        "RightsPrice",
    ]
    for col in numeric_columns:
        if col not in frame.columns:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "Amount" in frame.columns:
        frame["Amount"] = pd.to_numeric(frame["Amount"], errors="coerce")

    frame = frame.dropna(subset=required_columns)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)

    return frame.sort_index()


def fetch_yfinance_ohlcv(symbol: str, start_date: str, end_date: str, interval: str) -> DataSourceResult:
    yfinance_symbol = to_yfinance_symbol(symbol)
    inclusive_end = (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()
    data = yf.Ticker(yfinance_symbol).history(
        start=start_date,
        end=inclusive_end,
        interval=interval,
        auto_adjust=False,
        actions=True,
    )
    if data.empty:
        raise ValueError("yfinance returned empty data")
    warnings = []
    if yfinance_symbol != symbol:
        warnings.append(f"已将 {symbol} 转换为 yfinance 代码 {yfinance_symbol}。")
    return DataSourceResult(data=prepare_ohlcv(data), provider="yfinance", warnings=warnings)


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
        "fqt": "0",
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
        warnings=["东方财富不复权日 K 线；公司行动由双价格层统一处理。"],
    )


def fetch_mootdx_ohlcv(
    symbol: NormalizedSymbol,
    interval: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> DataSourceResult:
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
    page_size = 800
    pages = []
    target_start = pd.to_datetime(start_date) if start_date else None
    for page_start in range(0, page_size * 10, page_size):
        data_page = client.bars(symbol=symbol.code, frequency=frequency, start=page_start, offset=page_size)
        if data_page is None or data_page.empty:
            break
        pages.append(data_page)
        if target_start is None or len(data_page) < page_size:
            break
        page_index = pd.to_datetime(data_page.index)
        if page_index.min().normalize() <= target_start.normalize():
            break

    if not pages:
        raise ValueError("mootdx returned empty data")

    data = pd.concat(pages).sort_index()
    data = data[~data.index.duplicated(keep="first")]
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
    if "Volume" in data.columns and "volume" in data.columns:
        data = data.drop(columns=["volume"])
    if "datetime" in data.columns:
        data = data.set_index(pd.to_datetime(data["datetime"]))
    data = prepare_ohlcv(data)
    if start_date:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data.index < (pd.to_datetime(end_date) + pd.Timedelta(days=1))]
    if data.empty:
        raise ValueError("mootdx 在指定时间区间无数据")

    return DataSourceResult(
        data=data[["Open", "High", "Low", "Close", "Volume", "Amount"]],
        provider="mootdx",
        warnings=["mootdx bars 返回不复权原始价，跨除权除息日回测需谨慎。"],
    )


def fetch_mootdx_corporate_actions(symbol: NormalizedSymbol) -> pd.DataFrame:
    try:
        from mootdx.utils.adjust import get_xdxr
    except ImportError as exc:
        raise ValueError("mootdx is not installed") from exc
    actions = get_xdxr(symbol.code)
    if actions is None:
        raise ValueError("mootdx xdxr returned no result")
    return actions


def load_mootdx_corporate_actions_cache(
    symbol: NormalizedSymbol,
) -> tuple[pd.DataFrame, date] | None:
    path = _mootdx_corporate_actions_cache_path(symbol)
    if not path.exists():
        return None
    try:
        actions = pd.read_pickle(path)
        if not isinstance(actions, pd.DataFrame):
            return None
        refreshed_date = datetime.fromtimestamp(path.stat().st_mtime).date()
        return actions, refreshed_date
    except (OSError, ValueError, TypeError, EOFError):
        return None


def _mootdx_corporate_actions_cache_path(symbol: NormalizedSymbol) -> Path:
    from mootdx.config import get_config_path

    return Path(get_config_path(f"xdxr/{symbol.code}.plk"))


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
    *,
    prefer_cached_tail: bool = False,
) -> DataSourceResult:
    normalized = normalize_symbol(symbol)
    if normalized.market != "CN":
        raise ValueError("仅支持 A 股代码，请输入 6 位 A 股代码或 SH/SZ/BJ 前缀代码")

    if interval != "1d":
        return _fetch_live_ohlcv(symbol, normalized, start_date, end_date, interval, provider)

    def load_raw(selected_provider: str) -> DataSourceResult:
        if daily_cache_enabled():
            return _fetch_cached_daily_ohlcv(
                symbol,
                normalized,
                start_date,
                end_date,
                selected_provider,
                prefer_cached_tail=prefer_cached_tail,
            )
        return _fetch_live_ohlcv(
            symbol,
            normalized,
            start_date,
            end_date,
            interval,
            selected_provider,
        )

    raw_result = load_raw(provider)
    try:
        return _finalize_daily_result(
            raw_result,
            normalized,
            prefer_cached_corporate_actions=prefer_cached_tail,
        )
    except Exception as exc:
        if provider.lower() != "auto" or raw_result.provider != "mootdx":
            raise
        fallback = load_raw("yfinance")
        finalized = _finalize_daily_result(
            fallback,
            normalized,
            prefer_cached_corporate_actions=prefer_cached_tail,
        )
        finalized.warnings = [f"mootdx 除权数据获取失败: {exc}"] + finalized.warnings
        return finalized


def _finalize_daily_result(
    result: DataSourceResult,
    symbol: NormalizedSymbol,
    *,
    prefer_cached_corporate_actions: bool = False,
) -> DataSourceResult:
    warnings = list(result.warnings)
    warnings = [warning for warning in warnings if warning != YFINANCE_RAW_CACHE_MARKER]
    corporate_action_cache_status = result.corporate_action_cache_status
    if result.provider in {"mootdx", "eastmoney"}:
        actions, action_warnings, corporate_action_cache_status = (
            _resolve_mootdx_corporate_actions(
                result.data,
                symbol,
                prefer_cached=prefer_cached_corporate_actions,
            )
        )
        data = apply_corporate_actions(result.data, actions)
        warnings = [
            warning
            for warning in warnings
            if "不复权原始价" not in warning
        ]
        warnings.extend(action_warnings)
        warnings.append(
            "已启用双价格体系：复权价用于信号与图表，"
            f"{result.provider} 原始价用于成交与账户核算。"
        )
    elif result.provider == "yfinance":
        data = apply_corporate_actions(result.data, _yfinance_corporate_actions(result.data))
        corporate_action_cache_status = "provider_fallback"
        warnings.append("已启用双价格体系；yfinance 公司行动数据仅作为 mootdx 不可用时的备用。")
    else:
        data = apply_corporate_actions(result.data, None)
        warnings.append("该数据源未提供原始价与公司行动的完整拆分，成交价按其返回行情近似处理。")
    discontinuities = detect_unexplained_discontinuities(data)
    if discontinuities:
        first = discontinuities[0]
        warnings.append(
            "复权后检测到大幅波动，请结合上市/复牌状态核验: "
            f"{first['date']} {first['change_pct']:+.2f}%"
        )
    return DataSourceResult(
        data=data,
        provider=result.provider,
        warnings=warnings,
        cache_hit=result.cache_hit,
        cache_status=result.cache_status,
        corporate_action_cache_status=corporate_action_cache_status,
    )


def _resolve_mootdx_corporate_actions(
    data: pd.DataFrame,
    symbol: NormalizedSymbol,
    *,
    prefer_cached: bool,
) -> tuple[pd.DataFrame, list[str], str]:
    if not prefer_cached:
        return fetch_mootdx_corporate_actions(symbol), [], "refreshed"

    cached = load_mootdx_corporate_actions_cache(symbol)
    last_kline_date = pd.Timestamp(data.index.max()).date()
    if cached is not None:
        cached_actions, refreshed_date = cached
        if refreshed_date >= last_kline_date:
            return (
                cached_actions,
                [
                    "组合除权缓存复用："
                    f"缓存刷新至 {refreshed_date.isoformat()}，"
                    f"覆盖行情最后交易日 {last_kline_date.isoformat()}，"
                    "未触发逐股联网刷新。"
                ],
                "cache_reused",
            )

    try:
        return fetch_mootdx_corporate_actions(symbol), [], "refreshed"
    except Exception as exc:
        if cached is None:
            raise
        cached_actions, refreshed_date = cached
        return (
            cached_actions,
            [
                "mootdx 除权刷新失败，使用旧缓存；"
                f"除权缓存仅刷新至 {refreshed_date.isoformat()}，"
                f"行情截至 {last_kline_date.isoformat()}，"
                f"可能缺少近期公司行动: {exc}"
            ],
            "stale_fallback",
        )


def _yfinance_corporate_actions(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    dividends = (
        pd.to_numeric(data["Dividends"], errors="coerce").fillna(0.0)
        if "Dividends" in data.columns
        else pd.Series(0.0, index=data.index)
    )
    splits = (
        pd.to_numeric(data["Stock Splits"], errors="coerce").fillna(0.0)
        if "Stock Splits" in data.columns
        else pd.Series(0.0, index=data.index)
    )
    active = (dividends != 0) | (splits != 0)
    if not active.any():
        return pd.DataFrame()
    split_ratios = splits.where(splits > 0, 1.0)
    return pd.DataFrame(
        {
            "category": 1,
            "fenhong": dividends[active] * 10,
            "songzhuangu": (split_ratios[active] - 1) * 10,
            "peigu": 0.0,
            "peigujia": 0.0,
        },
        index=pd.DatetimeIndex(data.index[active]),
    )


def _fetch_cached_daily_ohlcv(
    symbol: str,
    normalized: NormalizedSymbol,
    start_date: str,
    end_date: str,
    provider: str,
    *,
    prefer_cached_tail: bool = False,
) -> DataSourceResult:
    normalized_provider = provider.lower()
    candidates = _cache_provider_candidates(normalized_provider)
    partial_snapshot = None
    for provider_name in candidates:
        snapshot = load_daily_cache(normalized.symbol, provider_name)
        if snapshot is None:
            continue
        if (
            snapshot.provider == "yfinance"
            and YFINANCE_RAW_CACHE_MARKER not in snapshot.warnings
        ):
            continue
        cached = slice_daily_cache(snapshot, start_date, end_date)
        if snapshot.covers(start_date, end_date) and not cached.empty:
            return DataSourceResult(
                data=prepare_ohlcv(cached),
                provider=snapshot.provider,
                warnings=list(snapshot.warnings),
                cache_hit=True,
                cache_status="hit",
            )
        gaps = uncovered_date_ranges(snapshot.covered_ranges, start_date, end_date)
        if (
            prefer_cached_tail
            and not cached.empty
            and _is_recent_trailing_gap(gaps, start_date, end_date)
        ):
            last_available_date = pd.Timestamp(cached.index.max()).date().isoformat()
            return DataSourceResult(
                data=prepare_ohlcv(cached),
                provider=snapshot.provider,
                warnings=[
                    *snapshot.warnings,
                    (
                        f"组合缓存优先模式：请求截至 {end_date}，"
                        f"{snapshot.provider} 行情仅截至 {last_available_date}；"
                        "使用最后可用缓存交易日，未触发逐股联网补全。"
                    ),
                ],
                cache_hit=True,
                cache_status="stale",
            )
        if partial_snapshot is None:
            partial_snapshot = snapshot

    if partial_snapshot is not None:
        try:
            fetched_frames = []
            fetched_warnings = list(partial_snapshot.warnings)
            gaps = uncovered_date_ranges(
                partial_snapshot.covered_ranges,
                start_date,
                end_date,
            )
            for gap_start, gap_end in gaps:
                gap_days = pd.date_range(gap_start, gap_end, freq="D")
                if len(gap_days) and all(day.weekday() >= 5 for day in gap_days):
                    continue
                result = _fetch_provider_ohlcv(
                    symbol,
                    normalized,
                    gap_start.isoformat(),
                    gap_end.isoformat(),
                    "1d",
                    partial_snapshot.provider,
                )
                fetched_frames.append(result.data)
                fetched_warnings.extend(result.warnings)
            snapshot = save_daily_cache(
                normalized.symbol,
                partial_snapshot.provider,
                pd.concat(fetched_frames) if fetched_frames else partial_snapshot.data,
                gaps,
                fetched_warnings,
            )
            cached = slice_daily_cache(snapshot, start_date, end_date)
            if not cached.empty:
                return DataSourceResult(
                    data=prepare_ohlcv(cached),
                    provider=snapshot.provider,
                    warnings=list(snapshot.warnings),
                    cache_hit=False,
                    cache_status="extended",
                )
        except Exception:
            pass

    live = _fetch_live_ohlcv(symbol, normalized, start_date, end_date, "1d", provider)
    snapshot = save_daily_cache(
        normalized.symbol,
        live.provider,
        live.data,
        [(date.fromisoformat(start_date), date.fromisoformat(end_date))],
        [
            *live.warnings,
            *(
                [YFINANCE_RAW_CACHE_MARKER]
                if live.provider == "yfinance"
                else []
            ),
        ],
    )
    cached = slice_daily_cache(snapshot, start_date, end_date)
    return DataSourceResult(
        data=prepare_ohlcv(cached),
        provider=live.provider,
        warnings=list(live.warnings),
        cache_hit=False,
        cache_status="miss",
    )


def _cache_provider_candidates(provider: str) -> list[str]:
    if provider == "auto":
        return ["mootdx", "yfinance"]
    if provider in {"mootdx", "yfinance", "eastmoney"}:
        return [provider]
    return []


def _is_recent_trailing_gap(
    gaps: list[tuple[date, date]],
    start_date: str,
    end_date: str,
) -> bool:
    if len(gaps) != 1:
        return False
    requested_start = date.fromisoformat(start_date)
    requested_end = date.fromisoformat(end_date)
    gap_start, gap_end = gaps[0]
    return (
        gap_start > requested_start
        and gap_end == requested_end
        and (gap_end - gap_start).days + 1 <= PORTFOLIO_CACHE_MAX_STALE_DAYS
    )


def _fetch_provider_ohlcv(
    symbol: str,
    normalized: NormalizedSymbol,
    start_date: str,
    end_date: str,
    interval: str,
    provider: str,
) -> DataSourceResult:
    if provider == "mootdx":
        return fetch_mootdx_ohlcv(normalized, interval, start_date, end_date)
    if provider == "eastmoney" and interval == "1d":
        return fetch_eastmoney_daily_ohlcv(normalized, start_date, end_date)
    if provider == "yfinance":
        return fetch_yfinance_ohlcv(symbol, start_date, end_date, interval)
    raise ValueError(f"不支持的数据源或市场组合: provider={provider}, symbol={symbol}, interval={interval}")


def _fetch_live_ohlcv(
    symbol: str,
    normalized: NormalizedSymbol,
    start_date: str,
    end_date: str,
    interval: str,
    provider: str,
) -> DataSourceResult:

    normalized_provider = provider.lower()
    attempts: list[tuple[str, Callable[[], DataSourceResult]]] = []

    if normalized_provider in ("auto", "mootdx") and normalized.market == "CN":
        attempts.append(("mootdx", lambda: _fetch_provider_ohlcv(symbol, normalized, start_date, end_date, interval, "mootdx")))
    if normalized_provider == "eastmoney" and normalized.market == "CN" and interval == "1d":
        attempts.append(("eastmoney", lambda: _fetch_provider_ohlcv(symbol, normalized, start_date, end_date, interval, "eastmoney")))
    if normalized_provider in ("auto", "yfinance"):
        attempts.append(("yfinance", lambda: _fetch_provider_ohlcv(symbol, normalized, start_date, end_date, interval, "yfinance")))

    errors: list[tuple[str, str]] = []
    for provider_name, attempt in attempts:
        try:
            result = attempt()
            if not result.data.empty:
                if errors:
                    fallback_warnings = [f"{name} 获取失败: {error}" for name, error in errors]
                    result.warnings = fallback_warnings + result.warnings
                return result
        except Exception as exc:
            errors.append((provider_name, str(exc)))

    if not attempts:
        raise ValueError(f"不支持的数据源或市场组合: provider={provider}, symbol={symbol}, interval={interval}")
    raise ValueError("所有数据源均获取失败: " + " | ".join(error for _, error in errors))
