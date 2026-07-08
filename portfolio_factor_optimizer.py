from __future__ import annotations

from itertools import product

from portfolio_factor_optimization_models import (
    PortfolioFactorCandidate,
    PortfolioFactorOptimizationRequest,
)
from portfolio_models import FactorConfig, SelectionConfig


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
