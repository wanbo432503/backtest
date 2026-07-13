from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

import pandas as pd

from market_data import fetch_ohlcv, prepare_ohlcv


BASE_RUNNER_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DUAL_PRICE_COLUMNS = [
    "RawOpen",
    "RawHigh",
    "RawLow",
    "RawClose",
    "AdjFactor",
]
CORPORATE_ACTION_COLUMNS = [
    "CashDividendPer10",
    "BonusSharesPer10",
    "RightsSharesPer10",
    "RightsPrice",
]


@dataclass
class PortfolioDataBundle:
    data_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0


def load_portfolio_ohlcv(
    symbols: list[str],
    start_date: str,
    end_date: str,
    provider: str = "auto",
    interval: str = "1d",
    min_history_bars: int = 0,
    batch_size: int = 20,
    batch_delay_seconds: float = 0.0,
    request_delay_seconds: float = 0.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> PortfolioDataBundle:
    if interval != "1d":
        raise ValueError("组合回测 MVP 仅支持日线")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if batch_delay_seconds < 0 or request_delay_seconds < 0:
        raise ValueError("rate limit delays must be non-negative")

    data_by_symbol: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    providers: dict[str, str] = {}
    failed_count = 0
    cache_hits = 0
    cache_misses = 0

    for batch_index, batch in enumerate(_chunks(symbols, batch_size)):
        for symbol_index, symbol in enumerate(batch):
            try:
                source_result = fetch_ohlcv(symbol, start_date, end_date, interval, provider)
                data = _prepare_runner_frame(source_result.data)
                if min_history_bars and len(data) < min_history_bars:
                    warnings.append(
                        f"{symbol} insufficient_history: {len(data)} bars < {min_history_bars}"
                    )
                    failed_count += 1
                else:
                    data_by_symbol[symbol] = data
                    providers[symbol] = source_result.provider
                    warnings.extend(source_result.warnings)
                    if source_result.cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1
            except Exception as exc:
                failed_count += 1
                warnings.append(f"{symbol} 获取失败: {exc}")
            _emit_load_progress(
                progress_callback,
                symbols,
                data_by_symbol,
                failed_count,
                symbol,
                cache_hits,
                cache_misses,
            )

            if request_delay_seconds and symbol_index < len(batch) - 1:
                sleeper(request_delay_seconds)

        if batch_delay_seconds and batch_index < _batch_count(symbols, batch_size) - 1:
            sleeper(batch_delay_seconds)

    if not data_by_symbol:
        raise ValueError("所有组合标的数据源均获取失败: " + "；".join(warnings))

    return PortfolioDataBundle(
        data_by_symbol=data_by_symbol,
        warnings=warnings,
        providers=providers,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def _chunks(symbols: list[str], batch_size: int) -> list[list[str]]:
    return [symbols[index:index + batch_size] for index in range(0, len(symbols), batch_size)]


def _batch_count(symbols: list[str], batch_size: int) -> int:
    if not symbols:
        return 0
    return (len(symbols) + batch_size - 1) // batch_size


def _emit_load_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    symbols: list[str],
    data_by_symbol: dict[str, pd.DataFrame],
    failed_count: int,
    current_symbol: str,
    cache_hits: int,
    cache_misses: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback({
        "phase": "loading_ohlcv",
        "total_symbols": len(symbols),
        "loaded_count": len(data_by_symbol),
        "failed_count": failed_count,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "current_symbol": current_symbol,
    })


def _prepare_runner_frame(data: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_ohlcv(data)
    columns = list(BASE_RUNNER_COLUMNS)
    if any(column in prepared.columns for column in DUAL_PRICE_COLUMNS):
        required = [*DUAL_PRICE_COLUMNS, *CORPORATE_ACTION_COLUMNS]
        missing = [column for column in required if column not in prepared.columns]
        if missing:
            raise ValueError(f"双价格行情缺少必要列: {missing}")
        columns.extend(required)
    frame = prepared[columns].copy()
    frame.attrs = dict(prepared.attrs)
    frame.index = pd.to_datetime(frame.index)
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    if isinstance(frame.index, pd.DatetimeIndex):
        frame.index = frame.index.normalize()
    return frame
