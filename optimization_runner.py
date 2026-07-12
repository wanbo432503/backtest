from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from math import prod
from typing import Any, Callable, Iterable

from backtest_runner import BacktestResult, run_single_backtest
from optimization_models import OptimizationRequest, StrategyParamConfig
from strategy_library import StrategyLibrary, get_strategy_library


OVERFIT_SCORE_GAP = 5


@dataclass
class OptimizationResult:
    objective: str
    symbols: list[str]
    top_results: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    progress_log: list[str] = field(default_factory=list)

    def to_api_response(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "symbols": self.symbols,
            "top_results": self.top_results,
            "warnings": self.warnings,
            "progress_log": self.progress_log,
        }


def expand_search_space(search_space: dict[str, list[Any]], max_combinations: int) -> list[dict[str, Any]]:
    if not search_space:
        return [{}]

    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    total_combinations = prod(len(items) for items in values)
    if total_combinations <= max_combinations:
        return [dict(zip(keys, items)) for items in product(*values)]

    return [
        _combination_at_index(keys, values, index)
        for index in _evenly_spaced_indices(total_combinations, max_combinations)
    ]


def _evenly_spaced_indices(total_count: int, sample_count: int) -> list[int]:
    if sample_count <= 1:
        return [0]

    step = (total_count - 1) / (sample_count - 1)
    indices = []
    seen = set()
    for position in range(sample_count):
        index = round(position * step)
        while index in seen and index < total_count - 1:
            index += 1
        while index in seen and index > 0:
            index -= 1
        seen.add(index)
        indices.append(index)
    return indices


def _combination_at_index(
    keys: list[str],
    values: list[list[Any]],
    index: int,
) -> dict[str, Any]:
    selected = {}
    remaining = index
    for key, options in reversed(list(zip(keys, values))):
        selected[key] = options[remaining % len(options)]
        remaining //= len(options)
    return {key: selected[key] for key in keys}


def score_backtest_result(result: BacktestResult) -> float:
    return float(result.metrics.get("score", 0))


def run_optimization(
    request: OptimizationRequest,
    strategy_library: StrategyLibrary | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> OptimizationResult:
    library = strategy_library or get_strategy_library()
    config = request.optimization_config
    strategies = config.strategies or request.strategies
    warnings = []
    progress_log = []
    rows = []
    trial_groups = []
    trial_specs = []

    if not strategies:
        warnings.append("未配置优化策略")

    for symbol in config.symbols:
        for strategy_config in strategies:
            combinations = expand_search_space(
                strategy_config.search_space,
                max_combinations=config.max_combinations,
            )
            trial_groups.append((symbol, strategy_config, combinations))
            progress_log.append(
                f"{symbol} {strategy_config.strategy_name}: {len(combinations)} combinations"
            )
            for combination in combinations:
                params = {**strategy_config.fixed_params, **combination}
                trial_specs.append((symbol, strategy_config, params))

    total_trials = sum(len(combinations) for _, _, combinations in trial_groups)
    max_workers = max(1, min(config.max_workers, total_trials or 1))
    _emit_progress(
        progress_callback,
        phase="optimizing",
        message="正在并行回测参数候选",
        total_trials=total_trials,
        completed_trials=0,
        current_symbol=config.symbols[0] if config.symbols else None,
        current_strategy=strategies[0].strategy_name if strategies else None,
        max_workers=max_workers,
        best_validate_score=None,
    )
    completed_trials = 0

    for batch in _batched(trial_specs, max_workers):
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
            futures = {
                executor.submit(
                    run_train_validate,
                    symbol=symbol,
                    strategy_config=strategy_config,
                    params=params,
                    request=request,
                    strategy_library=library,
                ): (symbol, strategy_config)
                for symbol, strategy_config, params in batch
            }
            for future in as_completed(futures):
                symbol, strategy_config = futures[future]
                row = future.result()
                rows.append(_decorate_result(row, min_trades=config.min_trades))
                completed_trials += 1
                best_row = max(rows, key=_optimization_sort_key) if rows else None
                _emit_progress(
                    progress_callback,
                    phase="optimizing",
                    message="正在并行回测参数候选",
                    total_trials=total_trials,
                    completed_trials=completed_trials,
                    current_symbol=symbol,
                    current_strategy=strategy_config.strategy_name,
                    max_workers=max_workers,
                    best_validate_score=best_row["validate_score"] if best_row else None,
                )

    rows.sort(key=_optimization_sort_key, reverse=True)
    top_results = rows[: config.top_n]
    for index, row in enumerate(top_results, start=1):
        row["rank"] = index

    _emit_progress(
        progress_callback,
        phase="completed",
        message="参数优化完成",
        total_trials=total_trials,
        completed_trials=completed_trials,
        max_workers=max_workers,
        best_validate_score=top_results[0]["validate_score"] if top_results else None,
    )

    return OptimizationResult(
        objective=config.objective,
        symbols=config.symbols,
        top_results=top_results,
        warnings=warnings,
        progress_log=progress_log,
    )


def run_train_validate(
    symbol: str,
    strategy_config: StrategyParamConfig,
    params: dict[str, Any],
    request: OptimizationRequest,
    strategy_library: StrategyLibrary | None = None,
) -> dict[str, Any]:
    config = request.optimization_config
    train_result = run_single_backtest(
        symbol=symbol,
        start_date=config.train_start_date or request.start_date,
        end_date=config.train_end_date or request.end_date,
        interval=config.interval or request.interval,
        strategy_name=strategy_config.strategy_name,
        strategy_library=strategy_library,
        initial_cash=request.initial_cash,
        data_provider=config.data_provider or request.data_provider,
        strategy_params=params,
        min_trades=config.min_trades,
    )
    validate_result = run_single_backtest(
        symbol=symbol,
        start_date=config.validate_start_date or config.train_start_date or request.start_date,
        end_date=config.validate_end_date or request.end_date,
        interval=config.interval or request.interval,
        strategy_name=strategy_config.strategy_name,
        strategy_library=strategy_library,
        initial_cash=request.initial_cash,
        data_provider=config.data_provider or request.data_provider,
        strategy_params=params,
        min_trades=config.min_trades,
    )

    return {
        "symbol": symbol,
        "strategy_name": strategy_config.strategy_name,
        "params": params,
        "train_metrics": train_result.metrics,
        "validate_metrics": validate_result.metrics,
        "train_score": score_backtest_result(train_result),
        "validate_score": score_backtest_result(validate_result),
        "validate_stats": validate_result.stats,
        "data_provider": validate_result.data_provider,
        "data_warnings": validate_result.data_warnings,
    }


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    **event: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(event)


def _batched(items: Iterable[Any], batch_size: int) -> list[list[Any]]:
    batch_size = max(1, batch_size)
    batch = []
    batches = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def _optimization_sort_key(item: dict[str, Any]) -> tuple[int, float, float]:
    validate_metrics = item.get("validate_metrics", {})
    is_rankable = bool(validate_metrics.get("is_rankable"))
    return (
        1 if is_rankable else 0,
        float(item.get("validate_score", 0)),
        float(item.get("train_score", 0)),
    )


def _decorate_result(row: dict[str, Any], min_trades: int) -> dict[str, Any]:
    risk_flags = []
    validate_metrics = row.get("validate_metrics", {})
    if int(validate_metrics.get("trades", 0)) < min_trades:
        risk_flags.append("too_few_trades")
    if row.get("validate_score", 0) < 0:
        risk_flags.append("validation_score_negative")
    if row.get("train_score", 0) - row.get("validate_score", 0) > OVERFIT_SCORE_GAP:
        risk_flags.append("possible_overfit")
    if validate_metrics.get("is_high_risk"):
        risk_flags.append("high_drawdown")

    return {
        **row,
        "risk_flags": risk_flags,
        "recommended": not risk_flags,
    }
