from __future__ import annotations

from math import ceil
from typing import Any, Callable

import pandas as pd

from signal_portfolio_models import (
    SignalMarketFilterConfig,
    SignalPortfolioBacktestRequest,
    SignalPortfolioBacktestResult,
)
from strategy_library import StrategyLibrary, get_strategy_library
from strategy_engine import StrategyDefinition
from strategy_simulator import SimulationConfig, run_strategy_simulation
from universe_scan_runner import load_universe_scan_data


def run_signal_portfolio_backtest(
    request: SignalPortfolioBacktestRequest,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    strategy_library: StrategyLibrary | None = None,
) -> SignalPortfolioBacktestResult:
    library = strategy_library or get_strategy_library()
    normalized = request.normalized_for_library(library)
    definition = library.get(normalized.strategy.strategy_name)
    data_start_date = _portfolio_data_start_date(normalized.start_date, definition)
    load_request = normalized.model_copy(deep=True)
    load_request.start_date = data_start_date
    scan = load_universe_scan_data(load_request, progress_callback=progress_callback)
    scan.diagnostics.update(
        {
            "history_data_start_date": data_start_date,
            "entry_history_bars": definition.portfolio_priority_history_bars,
        }
    )
    return run_signal_portfolio_with_data(
        normalized,
        scan.data_by_symbol,
        providers=scan.providers,
        warnings=scan.warnings,
        diagnostics=scan.diagnostics,
        progress_callback=progress_callback,
        strategy_library=library,
    )


def run_signal_portfolio_with_data(
    request: SignalPortfolioBacktestRequest,
    data_by_symbol: dict[str, pd.DataFrame],
    *,
    providers: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    strategy_library: StrategyLibrary | None = None,
) -> SignalPortfolioBacktestResult:
    if not data_by_symbol:
        raise ValueError("多股票信号组合没有可用行情数据")

    library = strategy_library or get_strategy_library()
    normalized = request.normalized_for_library(library)
    definition = library.get(normalized.strategy.strategy_name)
    strategy_config = library.validate_config(
        definition.strategy_id,
        normalized.strategy.parameters,
    )
    _emit(progress_callback, phase="building_signals", screened_count=len(data_by_symbol))
    overlay, breadth_diagnostics = _build_market_breadth_overlay(
        data_by_symbol,
        normalized.market_filter,
    )

    def risk_multiplier(symbol, date, row, intent) -> float:
        del symbol, row, intent
        if date not in overlay.index:
            return 0.0 if normalized.market_filter.enabled else 1.0
        return float(overlay.loc[date, "risk_multiplier"])

    def forward_progress(event: dict[str, Any]) -> None:
        forwarded = dict(event)
        if forwarded.get("phase") == "backtesting":
            forwarded["phase"] = "signal_backtesting"
        _emit(progress_callback, **forwarded)

    simulation = run_strategy_simulation(
        definition,
        strategy_config,
        data_by_symbol,
        SimulationConfig(
            initial_cash=normalized.initial_cash,
            max_positions=normalized.risk.max_positions,
            max_position_pct=normalized.risk.max_position_pct,
            target_gross_exposure=normalized.risk.target_gross_exposure,
            max_drawdown_stop_pct=normalized.risk.max_drawdown_stop_pct,
            trading=normalized.trading,
            start_date=normalized.start_date,
            end_date=normalized.end_date,
            min_entry_history_bars=definition.portfolio_priority_history_bars,
        ),
        progress_callback=forward_progress,
        entry_risk_multiplier=risk_multiplier,
    )

    signal_events = []
    for event in simulation.signal_events:
        enriched = dict(event)
        date = pd.Timestamp(event["date"])
        if date in overlay.index:
            enriched["market_breadth_pct"] = _optional_round(
                overlay.loc[date, "breadth_pct"]
            )
            enriched["market_risk_multiplier"] = round(
                float(overlay.loc[date, "risk_multiplier"]),
                4,
            )
        signal_events.append(enriched)

    result_diagnostics = dict(diagnostics or {})
    result_diagnostics.update(simulation.diagnostics)
    result_diagnostics.update(breadth_diagnostics)
    result_diagnostics.update(
        {
            "loaded_symbols": len(data_by_symbol),
            "strategy_id": definition.strategy_id,
            "strategy_name": definition.display_name,
            "traded_symbols": len(
                {trade["symbol"] for trade in simulation.trades}
            ),
            "providers": providers or {},
            "breadth_blocked_signal_count": sum(
                event.get("market_risk_multiplier") == 0
                for event in signal_events
            ),
            "breadth_partial_signal_count": sum(
                0 < event.get("market_risk_multiplier", 1) < 1
                for event in signal_events
            ),
        }
    )
    return SignalPortfolioBacktestResult(
        summary=simulation.summary,
        equity_curve=simulation.equity_curve,
        positions=simulation.positions,
        trades=simulation.trades,
        symbol_contributions=simulation.symbol_contributions,
        signal_events=signal_events,
        data_warnings=list(warnings or []),
        risk_flags=_risk_flags(
            simulation.summary,
            warnings or [],
            bool(simulation.diagnostics.get("drawdown_entry_stop")),
        ),
        scan_diagnostics=result_diagnostics,
        config=normalized.model_dump(mode="json"),
    )


def _portfolio_data_start_date(
    start_date: str,
    definition: StrategyDefinition,
) -> str:
    history_bars = definition.portfolio_priority_history_bars
    if history_bars <= 0:
        return start_date
    warmup_months = ceil(history_bars / 250 * 12 * 1.2)
    return (
        pd.Timestamp(start_date) - pd.DateOffset(months=warmup_months)
    ).strftime("%Y-%m-%d")


def _build_market_breadth_overlay(
    data_by_symbol: dict[str, pd.DataFrame],
    config: SignalMarketFilterConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    closes: dict[str, pd.Series] = {}
    for symbol, data in data_by_symbol.items():
        close = data["Close"].astype(float).copy()
        close.index = pd.DatetimeIndex(close.index).normalize()
        closes[symbol] = close.sort_index()
    close_matrix = pd.DataFrame(closes).sort_index()

    moving_average = close_matrix.rolling(
        config.breadth_ma_period,
        min_periods=config.breadth_ma_period,
    ).mean()
    valid = moving_average.notna() & close_matrix.notna()
    breadth = (close_matrix > moving_average).where(valid).mean(axis=1, skipna=True)
    breadth_pct = breadth * 100

    if config.enabled:
        multiplier = pd.Series(0.0, index=close_matrix.index)
        partial = (
            (breadth_pct >= config.market_breadth_min_pct)
            & (breadth_pct <= config.market_breadth_threshold_pct)
        )
        full = breadth_pct > config.market_breadth_threshold_pct
        multiplier.loc[partial.fillna(False)] = (
            config.market_breadth_partial_risk_pct / 100
        )
        multiplier.loc[full.fillna(False)] = 1.0
    else:
        multiplier = pd.Series(1.0, index=close_matrix.index)

    overlay = pd.DataFrame(
        {
            "breadth_pct": breadth_pct,
            "sample_size": valid.sum(axis=1).astype(int),
            "risk_multiplier": multiplier,
        }
    )
    observed = breadth_pct.dropna()
    diagnostics = {
        "market_filter_enabled": config.enabled,
        "breadth_ma_period": config.breadth_ma_period,
        "market_breadth_min_pct": config.market_breadth_min_pct,
        "market_breadth_threshold_pct": config.market_breadth_threshold_pct,
        "market_breadth_partial_risk_pct": config.market_breadth_partial_risk_pct,
        "average_market_breadth_pct": (
            round(float(observed.mean()), 6) if not observed.empty else None
        ),
        "breadth_observation_days": int(len(observed)),
    }
    return overlay, diagnostics


def _risk_flags(summary, warnings, entry_blocked):
    flags = []
    if summary["trades"] < 5:
        flags.append("too_few_trades")
    if summary["max_drawdown_pct"] > 30:
        flags.append("high_drawdown")
    if summary["final_gross_exposure"] < 0.2:
        flags.append("underinvested")
    if warnings:
        flags.append("data_gaps")
    if entry_blocked:
        flags.append("drawdown_entry_stop")
    return flags


def _optional_round(value: Any) -> float | None:
    return None if pd.isna(value) else round(float(value), 4)


def _emit(callback, **event):
    if callback is not None:
        callback(event)
