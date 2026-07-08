from __future__ import annotations

from collections.abc import Callable
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import product
from typing import Any

from portfolio_factor_optimization_models import (
    PortfolioFactorCandidate,
    PortfolioFactorMetrics,
    PortfolioFactorOptimizationRequest,
    PortfolioFactorOptimizationTrialResult,
)
from portfolio_backtest_runner import PortfolioBacktestContext, run_portfolio_backtest_with_context
from portfolio_models import (
    FactorConfig,
    PortfolioBacktestRequest,
    PortfolioBacktestResult,
    SelectionConfig,
)


_REBALANCE_CYCLE_DAYS = {
    "weekly": 7,
    "biweekly": 14,
    "monthly": 31,
}
BacktestRunner = Callable[
    [PortfolioBacktestRequest, PortfolioBacktestContext],
    PortfolioBacktestResult,
]


@dataclass(frozen=True)
class ResolvedOptimizationSplit:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    train_request: PortfolioBacktestRequest
    validation_request: PortfolioBacktestRequest
    train_calendar_days: int
    validation_calendar_days: int
    rebalance_frequency: str
    minimum_window_days: int
    method: str
    train_ratio: float | None

    def to_result_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "train_ratio": self.train_ratio,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "train_calendar_days": self.train_calendar_days,
            "validation_calendar_days": self.validation_calendar_days,
            "rebalance_frequency": self.rebalance_frequency,
            "minimum_window_days": self.minimum_window_days,
        }


def generate_factor_candidates(
    request: PortfolioFactorOptimizationRequest,
) -> list[PortfolioFactorCandidate]:
    rows = []
    seen: set[tuple] = set()
    space = request.search_space
    for values in product(
        space.momentum_lookback,
        space.volatility_lookback,
        space.liquidity_lookback,
        space.momentum_weight,
        space.volatility_weight,
        space.liquidity_weight,
        space.trend_weight,
        space.top_n,
        space.score_threshold,
    ):
        key = _candidate_key(values)
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)

    candidates = []
    for index, key in enumerate(sorted(rows)[: request.max_trials], start=1):
        (
            momentum_lookback,
            volatility_lookback,
            liquidity_lookback,
            momentum_weight,
            volatility_weight,
            liquidity_weight,
            trend_weight,
            top_n,
            _score_sort,
            score_threshold,
        ) = key
        factor_payload = request.base_request.factors.model_dump()
        factor_payload.update({
            "momentum_lookback": momentum_lookback,
            "volatility_lookback": volatility_lookback,
            "liquidity_lookback": liquidity_lookback,
            "momentum_weight": momentum_weight,
            "volatility_weight": volatility_weight,
            "liquidity_weight": liquidity_weight,
            "trend_weight": trend_weight,
        })
        selection_payload = request.base_request.selection.model_dump()
        selection_payload.update({
            "top_n": top_n,
            "score_threshold": score_threshold,
        })
        candidates.append(
            PortfolioFactorCandidate(
                candidate_id=f"candidate-{index:04d}",
                factor_config=FactorConfig.model_validate(factor_payload),
                selection_config=SelectionConfig.model_validate(selection_payload),
            )
        )
    return candidates


def resolve_optimization_split(
    request: PortfolioFactorOptimizationRequest,
) -> ResolvedOptimizationSplit:
    base_request = request.base_request
    split_config = request.split
    start = _parse_date(base_request.start_date, "start_date")
    end = _parse_date(base_request.end_date, "end_date")
    total_days = (end - start).days
    if split_config.method == "date":
        validation_start = _parse_date(split_config.validation_start or "", "validation_start")
    else:
        validation_start = start + timedelta(days=int(total_days * split_config.train_ratio))

    if validation_start <= start:
        raise ValueError("validation_start must be after start_date")
    if validation_start >= end:
        raise ValueError("validation_start must be before end_date")

    train_end = validation_start - timedelta(days=1)
    train_days = (train_end - start).days
    validation_days = (end - validation_start).days
    minimum_window_days = _minimum_window_days(base_request.rebalance.frequency)
    if train_days < minimum_window_days:
        raise ValueError(
            "train period is too short for at least two rebalance cycles"
        )
    if validation_days < minimum_window_days:
        raise ValueError(
            "validation period is too short for at least two rebalance cycles"
        )

    train_request = _copy_request_with_dates(
        base_request,
        start,
        train_end,
        optimization_split="train",
    )
    validation_request = _copy_request_with_dates(
        base_request,
        validation_start,
        end,
        optimization_split="validation",
    )
    return ResolvedOptimizationSplit(
        train_start=_format_date(start),
        train_end=_format_date(train_end),
        validation_start=_format_date(validation_start),
        validation_end=_format_date(end),
        train_request=train_request,
        validation_request=validation_request,
        train_calendar_days=train_days,
        validation_calendar_days=validation_days,
        rebalance_frequency=base_request.rebalance.frequency,
        minimum_window_days=minimum_window_days,
        method=split_config.method,
        train_ratio=split_config.train_ratio if split_config.method == "ratio" else None,
    )


def calculate_equity_curve_quality(
    summary: dict[str, Any],
    equity_curve: list[dict[str, Any]],
) -> PortfolioFactorMetrics:
    equities = [
        float(point.get("equity", 0.0))
        for point in equity_curve
        if _is_positive_finite(point.get("equity", 0.0))
    ]
    returns = _equity_returns(equities)
    return_volatility_pct = _sample_std(returns) * math.sqrt(252) * 100
    downside_volatility_pct = _downside_deviation(returns) * math.sqrt(252) * 100
    log_equity_trend_r2 = _log_equity_trend_r2(equities)
    annual_return_pct = float(summary.get("annual_return_pct", 0.0) or 0.0)
    equity_trend_score = max(annual_return_pct, 0.0) * log_equity_trend_r2
    positive_return_day_ratio = (
        sum(1 for value in returns if value > 0) / len(returns)
        if returns
        else 0.0
    )
    return PortfolioFactorMetrics(
        final_equity=round(float(summary.get("final_equity", equities[-1] if equities else 0.0) or 0.0), 6),
        total_return_pct=round(float(summary.get("total_return_pct", 0.0) or 0.0), 6),
        annual_return_pct=round(annual_return_pct, 6),
        max_drawdown_pct=round(float(summary.get("max_drawdown_pct", 0.0) or 0.0), 6),
        turnover=round(float(summary.get("turnover", 0.0) or 0.0), 6),
        rebalances=int(summary.get("rebalances", 0) or 0),
        trades=int(summary.get("trades", 0) or 0),
        return_volatility_pct=round(return_volatility_pct, 6),
        downside_volatility_pct=round(downside_volatility_pct, 6),
        log_equity_trend_r2=round(log_equity_trend_r2, 6),
        equity_trend_score=round(equity_trend_score, 6),
        positive_return_day_ratio=round(positive_return_day_ratio, 6),
    )


def calculate_validation_smooth_uptrend_score(
    train_metrics: PortfolioFactorMetrics,
    validation_metrics: PortfolioFactorMetrics,
) -> float:
    score = (
        validation_metrics.equity_trend_score * 0.60
        + validation_metrics.annual_return_pct * 0.40
        + min(train_metrics.annual_return_pct, validation_metrics.annual_return_pct) * 0.10
        - validation_metrics.return_volatility_pct * 0.35
        - validation_metrics.downside_volatility_pct * 0.25
        - abs(validation_metrics.max_drawdown_pct) * 0.25
        - validation_metrics.turnover * 0.02
    )
    return round(score, 6)


def build_optimization_risk_flags(
    train_metrics: PortfolioFactorMetrics,
    validation_metrics: PortfolioFactorMetrics,
    *,
    selected_symbols_count: int | None = None,
) -> list[str]:
    flags = []
    if validation_metrics.annual_return_pct < 0:
        flags.append("negative_validation_return")
    if train_metrics.annual_return_pct - validation_metrics.annual_return_pct > 20:
        flags.append("train_validation_gap")
    if validation_metrics.return_volatility_pct > 35:
        flags.append("high_validation_volatility")
    if validation_metrics.log_equity_trend_r2 < 0.45:
        flags.append("low_equity_trend_quality")
    if validation_metrics.max_drawdown_pct > 30:
        flags.append("high_validation_drawdown")
    if validation_metrics.turnover > 10:
        flags.append("high_turnover")
    if validation_metrics.rebalances < 2:
        flags.append("too_few_rebalances")
    if selected_symbols_count is not None and selected_symbols_count < 1:
        flags.append("too_few_selected_symbols")
    return flags


def evaluate_factor_candidate(
    candidate: PortfolioFactorCandidate,
    split: ResolvedOptimizationSplit,
    context: PortfolioBacktestContext,
    *,
    backtest_runner: BacktestRunner = run_portfolio_backtest_with_context,
) -> PortfolioFactorOptimizationTrialResult:
    train_request = _apply_candidate_to_request(split.train_request, candidate)
    validation_request = _apply_candidate_to_request(split.validation_request, candidate)
    train_result = backtest_runner(train_request, context)
    validation_result = backtest_runner(validation_request, context)
    train_metrics = calculate_equity_curve_quality(
        train_result.summary,
        train_result.equity_curve,
    )
    validation_metrics = calculate_equity_curve_quality(
        validation_result.summary,
        validation_result.equity_curve,
    )
    selected_symbols = validation_result.scan_diagnostics.get("selected_symbols")
    selected_symbols_count = len(selected_symbols) if isinstance(selected_symbols, list) else None
    risk_flags = build_optimization_risk_flags(
        train_metrics,
        validation_metrics,
        selected_symbols_count=selected_symbols_count,
    )
    objective_score = calculate_validation_smooth_uptrend_score(
        train_metrics,
        validation_metrics,
    )
    warnings = _dedupe_text([
        *train_result.data_warnings,
        *validation_result.data_warnings,
    ])
    return PortfolioFactorOptimizationTrialResult(
        candidate=candidate,
        objective_score=objective_score,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        risk_flags=risk_flags,
        warnings=warnings,
    )


def _candidate_key(values: tuple) -> tuple:
    (
        momentum_lookback,
        volatility_lookback,
        liquidity_lookback,
        momentum_weight,
        volatility_weight,
        liquidity_weight,
        trend_weight,
        top_n,
        score_threshold,
    ) = values
    score_sort = float("-inf") if score_threshold is None else float(score_threshold)
    return (
        int(momentum_lookback),
        int(volatility_lookback),
        int(liquidity_lookback),
        float(momentum_weight),
        float(volatility_weight),
        float(liquidity_weight),
        float(trend_weight),
        int(top_n),
        score_sort,
        None if score_threshold is None else float(score_threshold),
    )


def _is_positive_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric > 0 and math.isfinite(numeric)


def _equity_returns(equities: list[float]) -> list[float]:
    returns = []
    for previous, current in zip(equities, equities[1:]):
        if previous <= 0:
            continue
        returns.append(current / previous - 1)
    return returns


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _downside_deviation(returns: list[float]) -> float:
    if not returns:
        return 0.0
    downside_squares = [min(value, 0.0) ** 2 for value in returns]
    return math.sqrt(sum(downside_squares) / len(returns))


def _log_equity_trend_r2(equities: list[float]) -> float:
    if len(equities) < 2:
        return 0.0
    y_values = [math.log(value) for value in equities]
    x_values = list(range(len(y_values)))
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    ss_xx = sum((value - x_mean) ** 2 for value in x_values)
    if ss_xx == 0:
        return 0.0
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / ss_xx
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x_value for x_value in x_values]
    ss_total = sum((value - y_mean) ** 2 for value in y_values)
    if ss_total == 0:
        return 1.0
    ss_residual = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(y_values, fitted)
    )
    return max(0.0, min(1.0, 1 - ss_residual / ss_total))


def _copy_request_with_dates(
    base_request: PortfolioBacktestRequest,
    start: date,
    end: date,
    *,
    optimization_split: str,
) -> PortfolioBacktestRequest:
    payload = base_request.model_dump(mode="json")
    metadata = dict(payload.get("metadata") or {})
    metadata["optimization_split"] = optimization_split
    payload.update({
        "start_date": _format_date(start),
        "end_date": _format_date(end),
        "metadata": metadata,
    })
    return base_request.__class__.model_validate(payload)


def _apply_candidate_to_request(
    base_request: PortfolioBacktestRequest,
    candidate: PortfolioFactorCandidate,
) -> PortfolioBacktestRequest:
    payload = base_request.model_dump(mode="json")
    metadata = dict(payload.get("metadata") or {})
    metadata["optimization_candidate_id"] = candidate.candidate_id
    payload.update({
        "factors": candidate.factor_config.model_dump(mode="json"),
        "selection": candidate.selection_config.model_dump(mode="json"),
        "metadata": metadata,
    })
    return base_request.__class__.model_validate(payload)


def _dedupe_text(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _minimum_window_days(frequency: str) -> int:
    return max(_REBALANCE_CYCLE_DAYS[frequency] * 2, 30)


def _parse_date(value: str, field_name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format") from exc


def _format_date(value: date) -> str:
    return value.isoformat()
