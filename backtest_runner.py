from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.resources import CDN

from analytics import extract_core_metrics
from market_data import fetch_ohlcv, prepare_ohlcv
from optimization_models import AShareTradingConfig
from strategy_library import StrategyLibrary, get_strategy_library
from strategy_simulator import SimulationConfig, run_strategy_simulation


@dataclass
class BacktestResult:
    plot_html: str
    stats: dict[str, Any]
    metrics: dict[str, Any]
    symbol: str
    interval: str
    data_provider: str
    data_warnings: list[str]
    data_cache_status: str = "disabled"

    def to_api_response(self) -> dict[str, Any]:
        return {
            "plot_html": self.plot_html,
            "stats": self.stats,
            "symbol": self.symbol,
            "interval": self.interval,
            "data_provider": self.data_provider,
            "data_warnings": self.data_warnings,
            "data_cache_status": self.data_cache_status,
            "metrics": self.metrics,
        }


def run_single_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    strategy_name: str = "macd_volume_divergence_risk_control",
    strategy_library: StrategyLibrary | None = None,
    initial_cash: float = 10000,
    commission: float = 0.002,
    data_provider: str = "auto",
    strategy_params: dict[str, Any] | None = None,
    min_trades: int = 5,
    trading_config: AShareTradingConfig | None = None,
) -> BacktestResult:
    _validate_dates(start_date, end_date)
    library = strategy_library or get_strategy_library()
    definition = library.get(strategy_name)
    config = library.validate_config(strategy_name, strategy_params)

    source_result = fetch_ohlcv(symbol, start_date, end_date, interval, data_provider)
    data = source_result.data
    if data.empty:
        raise ValueError("无法获取数据，请检查股票代码和时间区间")
    data = prepare_ohlcv(data)
    if len(data) < 50:
        raise ValueError("数据点太少，无法进行有意义的回测")

    trading = trading_config or AShareTradingConfig(
        buy_commission_pct=commission * 100,
        sell_commission_pct=commission * 100,
        stamp_tax_pct=0,
        min_commission=0,
        slippage_pct=0,
    )
    simulation = run_strategy_simulation(
        definition,
        config,
        {symbol: data},
        SimulationConfig(
            initial_cash=initial_cash,
            max_positions=1,
            max_position_pct=1,
            target_gross_exposure=1,
            max_drawdown_stop_pct=None,
            trading=trading,
            start_date=start_date,
            end_date=end_date,
        ),
    )
    metrics = extract_core_metrics(simulation.summary, min_trades=min_trades)
    stats = _format_stats(simulation.summary, metrics)
    return BacktestResult(
        plot_html=_render_plot_html(data, simulation.equity_curve, simulation.trades),
        stats=stats,
        metrics=metrics,
        symbol=symbol,
        interval=interval,
        data_provider=source_result.provider,
        data_warnings=source_result.warnings,
        data_cache_status=source_result.cache_status,
    )


def _validate_dates(start_date: str, end_date: str) -> None:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if start >= end:
        raise ValueError("开始日期必须早于结束日期")


def _render_plot_html(
    data: pd.DataFrame,
    equity_curve: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> str:
    price_source = ColumnDataSource(
        {
            "date": pd.DatetimeIndex(data.index),
            "close": data["Close"].astype(float).to_list(),
        }
    )
    price_plot = figure(
        title="Unified Strategy Backtest — Price and Trades",
        x_axis_type="datetime",
        height=360,
        sizing_mode="stretch_width",
    )
    price_plot.line("date", "close", source=price_source, line_width=2, legend_label="Close")
    for side, color, marker in (("buy", "#198754", "triangle"), ("sell", "#dc3545", "inverted_triangle")):
        rows = [trade for trade in trades if trade["side"] == side]
        if not rows:
            continue
        source = ColumnDataSource(
            {
                "date": [pd.Timestamp(row["date"]) for row in rows],
                "price": [float(row["price"]) for row in rows],
            }
        )
        price_plot.scatter(
            "date",
            "price",
            source=source,
            marker=marker,
            size=12,
            color=color,
            legend_label=side.title(),
        )
    equity_plot = figure(
        title="Equity Curve",
        x_axis_type="datetime",
        height=260,
        sizing_mode="stretch_width",
        x_range=price_plot.x_range,
    )
    equity_plot.line(
        [pd.Timestamp(point["date"]) for point in equity_curve],
        [float(point["equity"]) for point in equity_curve],
        line_width=2,
        color="#0d6efd",
    )
    layout = column(price_plot, equity_plot, sizing_mode="stretch_width")
    return file_html(layout, CDN, "Unified Strategy Backtest")


def _format_stats(summary: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "策略收益率": f"{float(summary.get('total_return_pct', 0)):.2f}%",
        "最大回撤": f"{float(summary.get('max_drawdown_pct', 0)):.2f}%",
        "基准收益率": "0.00%",
        "持仓时间": f"{float(summary.get('final_gross_exposure', 0)) * 100:.2f}%",
        "年复合增长率": f"{float(summary.get('annual_return_pct', 0)):.2f}%",
        "交易次数": int(summary.get("trades", 0)),
        "交易胜率": f"{float(summary.get('win_rate_pct', 0)):.2f}%",
        "夏普比率": f"{float(summary.get('sharpe', 0)):.2f}",
        "综合评分": f"{float(metrics.get('score', 0)):.2f}",
    }
