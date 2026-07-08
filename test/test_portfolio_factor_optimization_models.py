import pytest
from pydantic import ValidationError

from portfolio_factor_optimization_models import (
    FactorSearchSpace,
    OptimizationSplitConfig,
    PortfolioFactorCandidate,
    PortfolioFactorMetrics,
    PortfolioFactorOptimizationRequest,
    PortfolioFactorOptimizationResult,
    PortfolioFactorOptimizationTrialResult,
    SelectionStrategySearchSpace,
)
from portfolio_models import FactorConfig, PortfolioBacktestRequest, SelectionConfig


def _base_request() -> PortfolioBacktestRequest:
    return PortfolioBacktestRequest(
        start_date="2024-01-01",
        end_date="2026-01-01",
        universe={"mode": "auto", "symbols": [], "max_scan_symbols": 20},
        selection={"top_n": 2, "min_history_bars": 60},
    )


def test_factor_search_space_defaults_are_finite_and_include_top_n_twenty():
    space = FactorSearchSpace()

    assert space.momentum_lookback
    assert space.volatility_lookback
    assert space.liquidity_lookback
    assert space.momentum_weight
    assert space.volatility_weight
    assert space.liquidity_weight
    assert space.trend_weight
    assert space.top_n[-1] == 20
    assert max(space.top_n) == 20


def test_factor_search_space_rejects_empty_lists_and_top_n_above_twenty():
    with pytest.raises(ValidationError, match="must not be empty"):
        FactorSearchSpace(momentum_lookback=[])

    with pytest.raises(ValidationError, match="top_n"):
        FactorSearchSpace(top_n=[2, 21])


def test_selection_strategy_search_space_validates_candidates():
    space = SelectionStrategySearchSpace(
        strategy_id="steady_low_vol_momentum",
        factor_lookbacks={"momentum_return": [40, 60]},
        factor_weights={"momentum_return": [0.2, 0.4]},
        top_n=[2, 5, 20],
        score_threshold=[None],
    )

    assert space.factor_lookbacks["momentum_return"] == [40, 60]
    assert space.factor_weights["momentum_return"] == [0.2, 0.4]

    with pytest.raises(ValidationError, match="top_n"):
        SelectionStrategySearchSpace(
            strategy_id="bad",
            factor_weights={"momentum_return": [0.2]},
            top_n=[21],
        )

    with pytest.raises(ValidationError, match="lookback"):
        SelectionStrategySearchSpace(
            strategy_id="bad",
            factor_lookbacks={"momentum_return": [0]},
            factor_weights={"momentum_return": [0.2]},
        )

    with pytest.raises(ValidationError, match="finite"):
        SelectionStrategySearchSpace(
            strategy_id="bad",
            factor_weights={"momentum_return": [float("inf")]},
        )


def test_optimization_request_validates_trials_workers_backend_and_objective():
    request = PortfolioFactorOptimizationRequest(base_request=_base_request())

    assert request.max_trials == 200
    assert request.max_workers == 8
    assert request.executor_backend == "process"
    assert request.objective == "validation_smooth_uptrend"

    with pytest.raises(ValidationError, match="max_trials"):
        PortfolioFactorOptimizationRequest(base_request=_base_request(), max_trials=0)

    with pytest.raises(ValidationError, match="max_workers"):
        PortfolioFactorOptimizationRequest(base_request=_base_request(), max_workers=9)

    with pytest.raises(ValidationError, match="executor_backend"):
        PortfolioFactorOptimizationRequest(base_request=_base_request(), executor_backend="gpu")


def test_optimization_split_validates_ratio_and_explicit_validation_start():
    assert OptimizationSplitConfig(train_ratio=0.7).method == "ratio"

    with pytest.raises(ValidationError, match="train_ratio"):
        OptimizationSplitConfig(train_ratio=0.49)

    with pytest.raises(ValidationError, match="train_ratio"):
        OptimizationSplitConfig(train_ratio=0.91)

    with pytest.raises(ValidationError, match="validation_start"):
        OptimizationSplitConfig(method="date")

    with pytest.raises(ValidationError, match="validation_start"):
        OptimizationSplitConfig(method="date", validation_start="2025/07/01")

    assert OptimizationSplitConfig(method="date", validation_start="2025-07-01").validation_start == "2025-07-01"


def test_factor_optimization_result_models_are_json_serializable():
    candidate = PortfolioFactorCandidate(
        candidate_id="candidate-1",
        factor_config=FactorConfig(momentum_weight=0.5),
        selection_config=SelectionConfig(top_n=2),
    )
    metrics = PortfolioFactorMetrics(
        final_equity=110000,
        total_return_pct=10,
        annual_return_pct=8,
        max_drawdown_pct=4,
        turnover=1.2,
        rebalances=12,
        trades=18,
        return_volatility_pct=6,
        downside_volatility_pct=3,
        log_equity_trend_r2=0.82,
        equity_trend_score=6.56,
        positive_return_day_ratio=0.55,
    )
    trial = PortfolioFactorOptimizationTrialResult(
        candidate=candidate,
        objective_score=5.4,
        train_metrics=metrics,
        validation_metrics=metrics,
        risk_flags=["low_equity_trend_quality"],
    )
    result = PortfolioFactorOptimizationResult(
        best_result=trial,
        top_results=[trial],
        split={"train_start": "2024-01-01", "validation_start": "2025-07-01"},
    )

    dumped = result.model_dump(mode="json")

    assert dumped["best_result"]["candidate"]["factor_config"]["momentum_weight"] == 0.5
    assert dumped["top_results"][0]["validation_metrics"]["log_equity_trend_r2"] == 0.82
