from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting._stats import compute_stats

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
    summary: dict[str, Any] = field(default_factory=dict)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    signal_events: list[dict[str, Any]] = field(default_factory=list)

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
            "summary": self.summary,
            "equity_curve": self.equity_curve,
            "positions": self.positions,
            "trades": self.trades,
            "signal_events": self.signal_events,
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
    summary = {
        **simulation.summary,
        "benchmark_return_pct": _benchmark_return_pct(data),
    }
    metrics = extract_core_metrics(summary, min_trades=min_trades)
    stats = _format_stats(summary, metrics)
    return BacktestResult(
        plot_html=_render_backtesting_plot_html(
            data,
            simulation.equity_curve,
            simulation.trades,
            initial_cash=initial_cash,
        ),
        stats=stats,
        metrics=metrics,
        symbol=symbol,
        interval=interval,
        data_provider=source_result.provider,
        data_warnings=source_result.warnings,
        data_cache_status=source_result.cache_status,
        summary=summary,
        equity_curve=simulation.equity_curve,
        positions=simulation.positions,
        trades=simulation.trades,
        signal_events=simulation.signal_events,
    )


def _validate_dates(start_date: str, end_date: str) -> None:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if start >= end:
        raise ValueError("开始日期必须早于结束日期")


def _benchmark_return_pct(data: pd.DataFrame) -> float:
    closes = pd.to_numeric(data["Close"], errors="coerce").dropna()
    if len(closes) < 2 or float(closes.iloc[0]) == 0:
        return 0.0
    return round((float(closes.iloc[-1]) / float(closes.iloc[0]) - 1) * 100, 6)


class _PlotOnlyStrategy(Strategy):
    def init(self) -> None:
        pass

    def next(self) -> None:
        pass


class _PlotResultStrategy:
    def __init__(self) -> None:
        self._indicators = []

    def __str__(self) -> str:
        return "Unified Strategy Backtest"


def _render_backtesting_plot_html(
    data: pd.DataFrame,
    equity_curve: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    *,
    initial_cash: float,
) -> str:
    plot_data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    plot_data.index = pd.DatetimeIndex(plot_data.index).normalize()
    plot_data = plot_data.sort_index()
    equity = _aligned_plot_equity(plot_data.index, equity_curve, initial_cash)
    plot_trades = _backtesting_trade_frame(plot_data, trades)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
        )
        results = compute_stats(
            plot_trades,
            equity.to_numpy(dtype="float64"),
            plot_data,
            _PlotResultStrategy(),
        )
    drawdown_duration = results["_equity_curve"]["DrawdownDuration"]
    if drawdown_duration.isna().all():
        results["_equity_curve"].iloc[
            0,
            results["_equity_curve"].columns.get_loc("DrawdownDuration"),
        ] = 0.0

    plot_cash = max(
        float(initial_cash),
        float(plot_data["Close"].max()) + 1,
    )
    native_backtest = Backtest(
        plot_data,
        _PlotOnlyStrategy,
        cash=plot_cash,
        commission=0,
        exclusive_orders=True,
    )
    with TemporaryDirectory(prefix="backtest-plot-") as directory:
        filename = Path(directory) / "single-stock-backtest.html"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Starting with pandas version 4.0",
            )
            native_backtest.plot(
                results=results,
                filename=str(filename),
                plot_equity=True,
                plot_pl=True,
                plot_volume=True,
                plot_trades=True,
                open_browser=False,
            )
        return filename.read_text(encoding="utf-8")


def _aligned_plot_equity(
    index: pd.DatetimeIndex,
    equity_curve: list[dict[str, Any]],
    initial_cash: float,
) -> pd.Series:
    if not equity_curve:
        return pd.Series(float(initial_cash), index=index, dtype="float64")
    values = pd.Series(
        [float(point["equity"]) for point in equity_curve],
        index=pd.DatetimeIndex(
            [pd.Timestamp(point["date"]).normalize() for point in equity_curve]
        ),
        dtype="float64",
    )
    values = values[~values.index.duplicated(keep="last")].sort_index()
    return values.reindex(index).ffill().fillna(float(initial_cash))


def _backtesting_trade_frame(
    data: pd.DataFrame,
    trades: list[dict[str, Any]],
) -> pd.DataFrame:
    columns = [
        "Size",
        "EntryBar",
        "ExitBar",
        "EntryPrice",
        "ExitPrice",
        "SL",
        "TP",
        "PnL",
        "Commission",
        "ReturnPct",
        "EntryTime",
        "ExitTime",
        "Duration",
        "Tag",
    ]
    rows: list[dict[str, Any]] = []
    pending_buy: dict[str, Any] | None = None
    for trade in trades:
        if trade.get("side") == "buy":
            pending_buy = trade
            continue
        if trade.get("side") != "sell" or pending_buy is None:
            continue
        rows.append(_plot_trade_row(data, pending_buy, trade))
        pending_buy = None

    if pending_buy is not None:
        final_date = data.index[-1]
        final_price = float(data.iloc[-1]["Close"])
        shares = int(pending_buy["shares"])
        entry_price = float(pending_buy["price"])
        rows.append(
            _plot_trade_row(
                data,
                pending_buy,
                {
                    "date": final_date,
                    "price": final_price,
                    "cost": 0.0,
                    "pnl": shares * (final_price - entry_price)
                    - float(pending_buy.get("cost") or 0),
                    "reason": "open_at_end",
                },
            )
        )
    return pd.DataFrame(rows, columns=columns)


def _plot_trade_row(
    data: pd.DataFrame,
    buy: dict[str, Any],
    sell: dict[str, Any],
) -> dict[str, Any]:
    entry_time = pd.Timestamp(buy["date"]).normalize()
    exit_time = pd.Timestamp(sell["date"]).normalize()
    entry_bar = _plot_bar_index(data.index, entry_time)
    exit_bar = _plot_bar_index(data.index, exit_time)
    shares = int(buy["shares"])
    entry_price = float(buy["price"])
    exit_price = float(sell["price"])
    pnl = float(
        sell.get("pnl")
        if sell.get("pnl") is not None
        else shares * (exit_price - entry_price)
    )
    entry_value = shares * entry_price + float(buy.get("cost") or 0)
    return {
        "Size": shares,
        "EntryBar": entry_bar,
        "ExitBar": exit_bar,
        "EntryPrice": entry_price,
        "ExitPrice": exit_price,
        "SL": np.nan,
        "TP": np.nan,
        "PnL": pnl,
        "Commission": float(buy.get("cost") or 0) + float(sell.get("cost") or 0),
        "ReturnPct": pnl / entry_value if entry_value else 0.0,
        "EntryTime": data.index[entry_bar],
        "ExitTime": data.index[exit_bar],
        "Duration": data.index[exit_bar] - data.index[entry_bar],
        "Tag": str(sell.get("reason") or "signal_exit"),
    }


def _plot_bar_index(index: pd.DatetimeIndex, timestamp: pd.Timestamp) -> int:
    location = index.get_indexer([timestamp])[0]
    if location < 0:
        raise ValueError(f"绘图成交日期不在行情数据中: {timestamp.date()}")
    return int(location)


def _format_stats(summary: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "策略收益率": f"{float(summary.get('total_return_pct', 0)):.2f}%",
        "最大回撤": f"{float(summary.get('max_drawdown_pct', 0)):.2f}%",
        "基准收益率": f"{float(summary.get('benchmark_return_pct', 0)):.2f}%",
        "持仓时间": f"{float(summary.get('exposure_time_pct', 0)):.2f}%",
        "年复合增长率": f"{float(summary.get('annual_return_pct', 0)):.2f}%",
        "交易次数": int(summary.get("trades", 0)),
        "交易胜率": f"{float(summary.get('win_rate_pct', 0)):.2f}%",
        "夏普比率": f"{float(summary.get('sharpe', 0)):.2f}",
        "综合评分": f"{float(metrics.get('score', 0)):.2f}",
    }
