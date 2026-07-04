import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backtesting import Backtest, Strategy

from analytics import extract_core_metrics
from market_data import fetch_ohlcv, prepare_ohlcv


@dataclass
class BacktestResult:
    plot_html: str
    stats: dict[str, Any]
    metrics: dict[str, Any]
    symbol: str
    interval: str
    data_provider: str
    data_warnings: list[str]

    def to_api_response(self) -> dict[str, Any]:
        return {
            "plot_html": self.plot_html,
            "stats": self.stats,
            "symbol": self.symbol,
            "interval": self.interval,
            "data_provider": self.data_provider,
            "data_warnings": self.data_warnings,
            "metrics": self.metrics,
        }


def run_single_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    strategy_name: str = "sma_cross",
    strategy_registry: dict[str, type[Strategy]] | None = None,
    initial_cash: float = 10000,
    commission: float = 0.002,
    data_provider: str = "auto",
) -> BacktestResult:
    _validate_dates(start_date, end_date)
    registry = strategy_registry or {}

    source_result = fetch_ohlcv(symbol, start_date, end_date, interval, data_provider)
    data = source_result.data
    if data.empty:
        raise ValueError("无法获取数据，请检查股票代码和时间区间")

    data = prepare_ohlcv(data)
    if len(data) < 50:
        raise ValueError("数据点太少，无法进行有意义的回测")

    strategy_class = registry.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"策略 '{strategy_name}' 不存在")

    bt = Backtest(data, strategy_class, cash=initial_cash, commission=commission)
    raw_stats = bt.run()
    plot_html = _render_plot_html(bt)
    metrics = extract_core_metrics(raw_stats)
    stats = _format_stats(raw_stats, metrics)

    return BacktestResult(
        plot_html=plot_html,
        stats=stats,
        metrics=metrics,
        symbol=symbol,
        interval=interval,
        data_provider=source_result.provider,
        data_warnings=source_result.warnings,
    )


def _validate_dates(start_date: str, end_date: str) -> None:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if start >= end:
        raise ValueError("开始日期必须早于结束日期")


def _render_plot_html(backtest: Backtest) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp_file:
        temp_filename = tmp_file.name

    try:
        backtest.plot(filename=temp_filename, open_browser=False)
        with open(temp_filename, "r", encoding="utf-8") as file:
            return file.read()
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def _format_stats(raw_stats, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "策略收益率": f"{raw_stats['Return [%]']:.2f}%",
        "最大回撤": f"{raw_stats['Max. Drawdown [%]']:.2f}%",
        "基准收益率": f"{raw_stats['Buy & Hold Return [%]']:.2f}%",
        "持仓时间": f"{raw_stats['Exposure Time [%]']:.2f}%",
        "年复合增长率": f"{raw_stats['CAGR [%]']:.2f}%",
        "交易次数": int(raw_stats["# Trades"]),
        "交易胜率": f"{raw_stats['Win Rate [%]']:.2f}%",
        "夏普比率": f"{raw_stats['Sharpe Ratio']:.2f}",
        "综合评分": f"{metrics['score']:.2f}",
    }
