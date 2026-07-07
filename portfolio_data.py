from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from market_data import fetch_ohlcv, prepare_ohlcv


@dataclass
class PortfolioDataBundle:
    data_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)


def load_portfolio_ohlcv(
    symbols: list[str],
    start_date: str,
    end_date: str,
    provider: str = "auto",
    interval: str = "1d",
    min_history_bars: int = 0,
) -> PortfolioDataBundle:
    if interval != "1d":
        raise ValueError("组合回测 MVP 仅支持日线")

    data_by_symbol: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    providers: dict[str, str] = {}

    for symbol in symbols:
        try:
            source_result = fetch_ohlcv(symbol, start_date, end_date, interval, provider)
            data = _prepare_runner_frame(source_result.data)
            if min_history_bars and len(data) < min_history_bars:
                warnings.append(
                    f"{symbol} insufficient_history: {len(data)} bars < {min_history_bars}"
                )
                continue

            data_by_symbol[symbol] = data
            providers[symbol] = source_result.provider
            warnings.extend(source_result.warnings)
        except Exception as exc:
            warnings.append(f"{symbol} 获取失败: {exc}")

    if not data_by_symbol:
        raise ValueError("所有组合标的数据源均获取失败: " + "；".join(warnings))

    return PortfolioDataBundle(
        data_by_symbol=data_by_symbol,
        warnings=warnings,
        providers=providers,
    )


def _prepare_runner_frame(data: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_ohlcv(data)
    columns = ["Open", "High", "Low", "Close", "Volume"]
    return prepared[columns].copy()
