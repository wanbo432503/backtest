from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import pandas as pd

from factor_engine import FUNDAMENTAL_FACTOR_KEYS, score_candidates, score_candidates_with_strategy
from portfolio_data import load_portfolio_ohlcv
from portfolio_fundamentals import load_portfolio_fundamentals
from portfolio_models import PortfolioBacktestRequest
from portfolio_selection_strategy_library import get_selection_strategy
from portfolio_selection_strategy_models import PortfolioSelectionStrategyDefinition
from selection_engine import build_trading_calendar, select_top_candidates
from stock_universe_provider import StockUniverseRecord, get_default_stock_universe
from tradable_universe import TradableUniversePolicy, validate_universe


FUNDAMENTAL_PREFETCH_MIN = 40
FUNDAMENTAL_PREFETCH_MAX = 120
FUNDAMENTAL_PREFETCH_PER_POSITION = 5


@dataclass
class UniverseScanData:
    data_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)
    providers: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    universe_records: list[StockUniverseRecord] = field(default_factory=list)


@dataclass
class UniverseScanResult:
    selected_symbols: list[str] = field(default_factory=list)
    ranking: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_api_response(self) -> dict[str, Any]:
        return {
            "selected_symbols": self.selected_symbols,
            "ranking": self.ranking,
            "scan_diagnostics": self.diagnostics,
            "warnings": self.warnings,
        }


def load_universe_scan_data(
    request: PortfolioBacktestRequest,
    data_loader: Callable[..., Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> UniverseScanData:
    _emit_progress(progress_callback, {"phase": "discovering_universe"})
    records, diagnostics, warnings = _resolve_universe_records(request)
    symbols = [record.symbol for record in records]
    if not symbols:
        raise ValueError("默认 60/00 股票池为空，无法执行组合扫描")

    active_loader = data_loader or load_portfolio_ohlcv
    bundle = active_loader(
        symbols,
        request.start_date,
        request.end_date,
        provider=request.data_provider,
        interval="1d",
        min_history_bars=request.selection.min_history_bars,
        batch_size=request.universe.ohlcv_batch_size,
        batch_delay_seconds=request.universe.ohlcv_batch_delay_seconds,
        request_delay_seconds=request.universe.ohlcv_request_delay_seconds,
        progress_callback=progress_callback,
    )
    filtered_data, prefilter_counts = _apply_prefilters(bundle.data_by_symbol, request)
    skipped_by_reason = Counter()
    skipped_by_reason.update(prefilter_counts)
    diagnostics.update({
        "scan_symbol_count": len(symbols),
        "loaded_count": len(bundle.data_by_symbol),
        "load_failed_count": max(len(symbols) - len(bundle.data_by_symbol), 0),
        "prefilter_skipped_count": sum(prefilter_counts.values()),
        "screened_count": len(filtered_data),
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "ohlcv_batch_size": request.universe.ohlcv_batch_size,
        "ohlcv_batch_delay_seconds": request.universe.ohlcv_batch_delay_seconds,
        "ohlcv_request_delay_seconds": request.universe.ohlcv_request_delay_seconds,
    })
    return UniverseScanData(
        data_by_symbol=filtered_data,
        providers=bundle.providers,
        warnings=[*warnings, *bundle.warnings],
        diagnostics=diagnostics,
        universe_records=records,
    )


def run_universe_scan(
    request: PortfolioBacktestRequest,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> UniverseScanResult:
    scan_data = load_universe_scan_data(request, progress_callback=progress_callback)
    diagnostics = dict(scan_data.diagnostics)
    if not scan_data.data_by_symbol:
        diagnostics.update({"scored_count": 0, "selected_count": 0})
        return UniverseScanResult(
            selected_symbols=[],
            ranking=[],
            diagnostics=diagnostics,
            warnings=[*scan_data.warnings, "no_symbols_after_prefilter"],
        )

    calendar = build_trading_calendar(scan_data.data_by_symbol)
    if not calendar:
        diagnostics.update({"scored_count": 0, "selected_count": 0})
        return UniverseScanResult(
            selected_symbols=[],
            ranking=[],
            diagnostics=diagnostics,
            warnings=[*scan_data.warnings, "empty_trading_calendar"],
        )

    as_of_date = _scan_as_of_date(calendar, request.end_date)
    _emit_progress(progress_callback, {
        "phase": "scoring",
        "screened_count": len(scan_data.data_by_symbol),
        "as_of_date": _date_str(as_of_date),
    })
    warnings = list(scan_data.warnings)
    selection_strategy = _named_selection_strategy(request)
    if selection_strategy is not None:
        diagnostics.update({
            "selection_strategy_id": selection_strategy.strategy_id,
            "selection_strategy_name": selection_strategy.name,
        })
    fundamentals_by_symbol = _fundamentals_for_scan(
        scan_data.data_by_symbol,
        as_of_date,
        request,
        selection_strategy=selection_strategy,
        warnings=warnings,
        diagnostics=diagnostics,
        progress_callback=progress_callback,
    )

    ranking = _score_candidates_for_request(
        scan_data.data_by_symbol,
        as_of_date,
        request,
        selection_strategy=selection_strategy,
        fundamentals_by_symbol=fundamentals_by_symbol,
    )
    selection = select_top_candidates(ranking, request.selection)
    candidate_warnings = _candidate_warnings(ranking)
    diagnostics.update({
        "as_of_date": _date_str(as_of_date),
        "scored_count": len([row for row in ranking if row.get("skip_reason") is None]),
        "selected_count": len(selection.selected),
    })
    return UniverseScanResult(
        selected_symbols=[row["symbol"] for row in selection.selected],
        ranking=ranking,
        diagnostics=diagnostics,
        warnings=[*warnings, *selection.warnings, *candidate_warnings],
    )


def _named_selection_strategy(
    request: PortfolioBacktestRequest,
) -> PortfolioSelectionStrategyDefinition | None:
    config = request.selection_strategy
    if config is None or not config.enabled:
        return None
    return get_selection_strategy(config.strategy_id)


def _fundamentals_for_scan(
    data_by_symbol: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    request: PortfolioBacktestRequest,
    *,
    selection_strategy: PortfolioSelectionStrategyDefinition | None,
    warnings: list[str],
    diagnostics: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]] | None:
    if selection_strategy is None or not _strategy_needs_fundamentals(selection_strategy):
        return None
    symbols = _prefilter_fundamental_symbols(data_by_symbol, as_of_date, request)
    prefetch_limit = _fundamental_prefetch_limit(request, len(data_by_symbol))
    as_of_text = _date_str(as_of_date)
    _emit_progress(progress_callback, {
        "phase": "loading_fundamentals",
        "screened_count": len(symbols),
        "total_screened_count": len(data_by_symbol),
        "prefiltered_count": len(symbols),
        "prefetch_limit": prefetch_limit,
        "as_of_date": as_of_text,
    })
    bundle = load_portfolio_fundamentals(
        symbols,
        data_provider="akshare",
        as_of_date=as_of_text,
        progress_callback=_fundamental_progress_forwarder(
            progress_callback,
            screened_count=len(data_by_symbol),
            prefiltered_count=len(symbols),
            prefetch_limit=prefetch_limit,
            as_of_date=as_of_text,
        ),
    )
    diagnostics.update(bundle.to_diagnostics())
    diagnostics.update({
        "fundamental_prefetch_limit": prefetch_limit,
        "fundamental_prefiltered_count": len(symbols),
        "fundamental_total_screened_count": len(data_by_symbol),
    })
    warnings.extend(_new_warnings(warnings, bundle.warnings))
    return bundle.values_by_symbol


def _prefilter_fundamental_symbols(
    data_by_symbol: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    request: PortfolioBacktestRequest,
) -> list[str]:
    limit = _fundamental_prefetch_limit(request, len(data_by_symbol))
    symbols = list(data_by_symbol)
    if len(symbols) <= limit:
        return symbols
    ranking = score_candidates(data_by_symbol, as_of_date, request.factors, request.selection)
    ranked_symbols = [
        row["symbol"]
        for row in ranking
        if row.get("skip_reason") is None
    ]
    return ranked_symbols[:limit]


def _fundamental_prefetch_limit(request: PortfolioBacktestRequest, screened_count: int) -> int:
    configured = request.metadata.get("fundamental_prefetch_limit")
    if configured is not None:
        try:
            limit = int(configured)
        except (TypeError, ValueError):
            limit = FUNDAMENTAL_PREFETCH_MAX
    else:
        limit = max(
            request.selection.top_n * FUNDAMENTAL_PREFETCH_PER_POSITION,
            FUNDAMENTAL_PREFETCH_MIN,
        )
        limit = min(limit, FUNDAMENTAL_PREFETCH_MAX)
    return max(1, min(limit, screened_count))


def _fundamental_progress_forwarder(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    screened_count: int,
    prefiltered_count: int,
    prefetch_limit: int,
    as_of_date: str,
) -> Callable[[dict[str, Any]], None] | None:
    if progress_callback is None:
        return None

    def forward(event: dict[str, Any]) -> None:
        _emit_progress(progress_callback, {
            **event,
            "screened_count": prefiltered_count,
            "total_screened_count": screened_count,
            "prefiltered_count": prefiltered_count,
            "prefetch_limit": prefetch_limit,
            "as_of_date": as_of_date,
        })

    return forward


def _strategy_needs_fundamentals(strategy: PortfolioSelectionStrategyDefinition) -> bool:
    return any(factor.key in FUNDAMENTAL_FACTOR_KEYS for factor in strategy.factors)


def _new_warnings(existing: list[str], incoming: list[str]) -> list[str]:
    seen = set(existing)
    return [warning for warning in incoming if warning not in seen]


def _score_candidates_for_request(
    data_by_symbol: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
    request: PortfolioBacktestRequest,
    *,
    selection_strategy: PortfolioSelectionStrategyDefinition | None,
    fundamentals_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if selection_strategy is None:
        return score_candidates(data_by_symbol, as_of_date, request.factors, request.selection)
    parameter_overrides = (
        request.selection_strategy.parameter_overrides
        if request.selection_strategy is not None
        else {}
    )
    return score_candidates_with_strategy(
        data_by_symbol,
        as_of_date,
        request.selection,
        selection_strategy,
        parameter_overrides=parameter_overrides,
        fundamentals_by_symbol=fundamentals_by_symbol,
    )


def _candidate_warnings(ranking: list[dict[str, Any]]) -> list[str]:
    seen = set()
    warnings = []
    for row in ranking:
        for warning in row.get("warnings", []):
            if warning not in seen:
                warnings.append(str(warning))
                seen.add(warning)
    return warnings


def _resolve_universe_records(
    request: PortfolioBacktestRequest,
) -> tuple[list[StockUniverseRecord], dict[str, Any], list[str]]:
    mode = getattr(request.universe, "mode", "manual")
    if mode == "manual":
        result = validate_universe(
            list(request.universe.symbols),
            policy=TradableUniversePolicy(max_symbols=request.universe.max_symbols),
        )
        blocking = [row.reason for row in result.rejected if row.reason != "duplicate_symbol"]
        if blocking:
            raise ValueError("; ".join(str(reason) for reason in blocking))
        records = [_record_for_symbol(symbol, "manual") for symbol in result.accepted_symbols]
        return records, {
            "mode": "manual",
            "universe_source": "manual",
            "total_universe_size": len(records),
        }, []

    universe = get_default_stock_universe(
        refresh=bool(request.universe.refresh_universe),
    )
    records = _apply_auto_symbol_overrides(universe.records, request)
    total_universe_size = len(records)
    records = _limit_records(records, request.universe.max_scan_symbols)
    return records, {
        "mode": "auto",
        "universe_source": universe.source,
        "total_universe_size": total_universe_size,
        "max_scan_symbols": request.universe.max_scan_symbols,
    }, universe.warnings


def _apply_auto_symbol_overrides(
    records: list[StockUniverseRecord],
    request: PortfolioBacktestRequest,
) -> list[StockUniverseRecord]:
    blacklist = set(request.universe.blacklist_symbols)
    whitelist = set(request.universe.whitelist_symbols)
    filtered = [record for record in records if record.symbol not in blacklist]
    if whitelist:
        filtered = [record for record in filtered if record.symbol in whitelist]
    return filtered


def _apply_prefilters(
    data_by_symbol: dict[str, pd.DataFrame],
    request: PortfolioBacktestRequest,
) -> tuple[dict[str, pd.DataFrame], Counter]:
    filtered: dict[str, pd.DataFrame] = {}
    skipped = Counter()
    for symbol, data in data_by_symbol.items():
        reason = _prefilter_skip_reason(data, request)
        if reason:
            skipped[reason] += 1
            continue
        filtered[symbol] = data
    return filtered, skipped


def _prefilter_skip_reason(data: pd.DataFrame, request: PortfolioBacktestRequest) -> str | None:
    selection = request.selection
    liquidity_lookback = max(getattr(request.factors, "liquidity_lookback", 20), 1)
    tail = data.tail(liquidity_lookback)

    min_avg_volume = getattr(selection, "min_avg_volume", None)
    if min_avg_volume is not None and float(tail["Volume"].mean()) < float(min_avg_volume):
        return "below_min_avg_volume"

    min_avg_turnover = getattr(selection, "min_avg_turnover_value", None)
    if min_avg_turnover is not None:
        avg_turnover = (tail["Close"].astype(float) * tail["Volume"].astype(float)).mean()
        if float(avg_turnover) < float(min_avg_turnover):
            return "below_min_avg_turnover"

    last_close = float(data["Close"].iloc[-1])
    min_price = getattr(selection, "min_price", None)
    if min_price is not None and last_close < float(min_price):
        return "below_min_price"

    max_price = getattr(selection, "max_price", None)
    if max_price is not None and last_close > float(max_price):
        return "above_max_price"

    max_gap_days = getattr(selection, "max_data_gap_days", None)
    if max_gap_days is not None:
        end = pd.Timestamp(request.end_date).normalize()
        last_date = pd.Timestamp(data.index.max()).normalize()
        if (end - last_date).days > int(max_gap_days):
            return "recent_data_gap"

    return None


def _record_for_symbol(symbol: str, source: str) -> StockUniverseRecord:
    exchange = symbol[:2]
    code = symbol[2:]
    return StockUniverseRecord(
        symbol=symbol,
        exchange=exchange,
        code_prefix=code[:2],
        source=source,
    )


def _limit_records(
    records: list[StockUniverseRecord],
    max_symbols: int | None,
) -> list[StockUniverseRecord]:
    if max_symbols is None:
        return records
    return records[:max(max_symbols, 0)]


def _scan_as_of_date(calendar: list[pd.Timestamp], end_date: str) -> pd.Timestamp:
    end = pd.Timestamp(end_date)
    eligible = [date for date in calendar if date <= end]
    return eligible[-1] if eligible else calendar[-1]


def _date_str(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if progress_callback is not None:
        progress_callback(event)
