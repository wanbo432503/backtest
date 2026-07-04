from dataclasses import dataclass, field
from itertools import product
from typing import Any

from backtesting import Strategy

from backtest_runner import BacktestResult, run_single_backtest
from optimization_models import OptimizationRequest, StrategyParamConfig


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
    combinations = []
    for items in product(*values):
        combinations.append(dict(zip(keys, items)))
        if len(combinations) >= max_combinations:
            break
    return combinations


def score_backtest_result(result: BacktestResult) -> float:
    return float(result.metrics.get("score", 0))


def run_optimization(
    request: OptimizationRequest,
    strategy_registry: dict[str, type[Strategy]] | None = None,
) -> OptimizationResult:
    config = request.optimization_config
    strategies = config.strategies or request.strategies
    warnings = []
    progress_log = []
    rows = []

    if not strategies:
        warnings.append("未配置优化策略")

    for symbol in config.symbols:
        for strategy_config in strategies:
            combinations = expand_search_space(
                strategy_config.search_space,
                max_combinations=config.max_combinations,
            )
            progress_log.append(
                f"{symbol} {strategy_config.strategy_name}: {len(combinations)} combinations"
            )
            for combination in combinations:
                params = {**strategy_config.fixed_params, **combination}
                row = run_train_validate(
                    symbol=symbol,
                    strategy_config=strategy_config,
                    params=params,
                    request=request,
                    strategy_registry=strategy_registry,
                )
                rows.append(_decorate_result(row, min_trades=config.min_trades))

    rows.sort(key=lambda item: item["validate_score"], reverse=True)
    top_results = rows[: config.top_n]
    for index, row in enumerate(top_results, start=1):
        row["rank"] = index

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
    strategy_registry: dict[str, type[Strategy]] | None = None,
) -> dict[str, Any]:
    config = request.optimization_config
    train_result = run_single_backtest(
        symbol=symbol,
        start_date=config.train_start_date or request.start_date,
        end_date=config.train_end_date or request.end_date,
        interval=config.interval or request.interval,
        strategy_name=strategy_config.strategy_name,
        strategy_registry=strategy_registry,
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
        strategy_registry=strategy_registry,
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
