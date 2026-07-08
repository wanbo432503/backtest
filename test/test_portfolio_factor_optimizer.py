from portfolio_factor_optimization_models import FactorSearchSpace, PortfolioFactorOptimizationRequest
from portfolio_factor_optimizer import generate_factor_candidates
from portfolio_models import PortfolioBacktestRequest


def _base_request() -> PortfolioBacktestRequest:
    return PortfolioBacktestRequest(
        start_date="2024-01-01",
        end_date="2026-01-01",
        universe={"mode": "auto", "symbols": [], "max_scan_symbols": 20},
        factors={
            "momentum_lookback": 60,
            "volatility_lookback": 20,
            "liquidity_lookback": 20,
            "momentum_weight": 0.45,
            "volatility_weight": -0.25,
            "liquidity_weight": 0.2,
            "trend_weight": 0.1,
        },
        selection={"top_n": 2, "min_history_bars": 60, "score_threshold": None},
    )


def _small_search_space(**overrides) -> FactorSearchSpace:
    payload = {
        "momentum_lookback": [20],
        "volatility_lookback": [10],
        "liquidity_lookback": [10],
        "momentum_weight": [0.2],
        "volatility_weight": [-0.1],
        "liquidity_weight": [0.3],
        "trend_weight": [0.1],
        "top_n": [2],
        "score_threshold": [None],
    }
    payload.update(overrides)
    return FactorSearchSpace(**payload)


def test_generate_factor_candidates_is_deterministic_and_capped():
    request = PortfolioFactorOptimizationRequest(
        base_request=_base_request(),
        search_space=_small_search_space(
            top_n=[2, 5],
            score_threshold=[None, 0.25],
        ),
        max_trials=3,
    )

    first = generate_factor_candidates(request)
    second = generate_factor_candidates(request)

    assert len(first) == 3
    assert [candidate.model_dump(mode="json") for candidate in first] == [
        candidate.model_dump(mode="json") for candidate in second
    ]
    assert [candidate.candidate_id for candidate in first] == [
        "candidate-0001",
        "candidate-0002",
        "candidate-0003",
    ]


def test_generate_factor_candidates_dedupes_equivalent_search_values():
    request = PortfolioFactorOptimizationRequest(
        base_request=_base_request(),
        search_space=_small_search_space(
            momentum_lookback=[20, 20],
            top_n=[2, 2],
            score_threshold=[None, None],
        ),
    )

    candidates = generate_factor_candidates(request)

    assert len(candidates) == 1
    assert candidates[0].factor_config.momentum_lookback == 20
    assert candidates[0].selection_config.top_n == 2


def test_generate_factor_candidates_returns_valid_factor_and_selection_configs():
    request = PortfolioFactorOptimizationRequest(
        base_request=_base_request(),
        search_space=_small_search_space(
            momentum_lookback=[90],
            volatility_lookback=[40],
            liquidity_lookback=[20],
            momentum_weight=[0.65],
            volatility_weight=[-0.5],
            liquidity_weight=[0.15],
            trend_weight=[0.2],
            top_n=[20],
            score_threshold=[0.1],
        ),
    )

    [candidate] = generate_factor_candidates(request)

    assert candidate.factor_config.momentum_lookback == 90
    assert candidate.factor_config.volatility_weight == -0.5
    assert candidate.selection_config.top_n == 20
    assert candidate.selection_config.score_threshold == 0.1
    assert candidate.selection_config.min_history_bars == 60
