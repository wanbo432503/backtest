import pytest

from portfolio_factor_optimization_models import FactorSearchSpace, PortfolioFactorOptimizationRequest
from portfolio_factor_optimizer import (
    build_optimization_risk_flags,
    calculate_equity_curve_quality,
    calculate_validation_smooth_uptrend_score,
    evaluate_factor_candidate,
    generate_factor_candidates,
    resolve_optimization_split,
)
from portfolio_backtest_runner import PortfolioBacktestContext
from portfolio_models import PortfolioBacktestRequest, PortfolioBacktestResult


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


def _curve(values: list[float]) -> list[dict[str, float | str]]:
    return [
        {"date": f"2024-01-{index + 1:02d}", "equity": value}
        for index, value in enumerate(values)
    ]


def _summary(
    *,
    final_equity: float,
    annual_return_pct: float,
    max_drawdown_pct: float = 0.0,
    turnover: float = 1.0,
    rebalances: int = 6,
    trades: int = 12,
) -> dict[str, float | int]:
    return {
        "final_equity": final_equity,
        "total_return_pct": final_equity / 100000 * 100 - 100,
        "annual_return_pct": annual_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "turnover": turnover,
        "rebalances": rebalances,
        "trades": trades,
    }


def _portfolio_result(
    *,
    final_equity: float,
    annual_return_pct: float,
    values: list[float],
    warnings: list[str] | None = None,
    selected_symbols: list[str] | None = None,
    max_drawdown_pct: float = 0.0,
    turnover: float = 1.0,
    rebalances: int = 6,
    trades: int = 12,
) -> PortfolioBacktestResult:
    return PortfolioBacktestResult(
        summary=_summary(
            final_equity=final_equity,
            annual_return_pct=annual_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            turnover=turnover,
            rebalances=rebalances,
            trades=trades,
        ),
        equity_curve=_curve(values),
        data_warnings=warnings or [],
        scan_diagnostics={"selected_symbols": selected_symbols or ["SH600000"]},
    )


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


def test_resolve_optimization_split_uses_default_ratio_without_overlap():
    request = PortfolioFactorOptimizationRequest(
        base_request=_base_request(),
    )

    split = resolve_optimization_split(request)

    assert split.train_start == "2024-01-01"
    assert split.train_end == "2025-05-25"
    assert split.validation_start == "2025-05-26"
    assert split.validation_end == "2026-01-01"
    assert split.train_request.start_date == "2024-01-01"
    assert split.train_request.end_date == "2025-05-25"
    assert split.validation_request.start_date == "2025-05-26"
    assert split.validation_request.end_date == "2026-01-01"
    assert split.train_request.factors == request.base_request.factors
    assert split.validation_request.selection == request.base_request.selection
    assert split.to_result_payload() == {
        "method": "ratio",
        "train_ratio": 0.7,
        "train_start": "2024-01-01",
        "train_end": "2025-05-25",
        "validation_start": "2025-05-26",
        "validation_end": "2026-01-01",
        "train_calendar_days": 510,
        "validation_calendar_days": 220,
        "rebalance_frequency": "monthly",
        "minimum_window_days": 62,
    }


def test_resolve_optimization_split_supports_explicit_validation_start():
    request = PortfolioFactorOptimizationRequest(
        base_request=_base_request(),
        split={"method": "date", "validation_start": "2025-07-01"},
    )

    split = resolve_optimization_split(request)

    assert split.train_end == "2025-06-30"
    assert split.validation_start == "2025-07-01"
    assert split.validation_end == "2026-01-01"


@pytest.mark.parametrize(
    ("base_request", "split_config", "message"),
    [
        (
            PortfolioBacktestRequest(
                start_date="2024-01-01",
                end_date="2024-04-01",
                universe={"mode": "manual", "symbols": ["SH600000", "SZ000001"]},
                selection={"top_n": 2, "min_history_bars": 20},
            ),
            {"method": "ratio", "train_ratio": 0.7},
            "validation period is too short",
        ),
        (
            _base_request(),
            {"method": "date", "validation_start": "2026-01-01"},
            "validation_start must be before end_date",
        ),
        (
            _base_request(),
            {"method": "date", "validation_start": "2024-01-15"},
            "train period is too short",
        ),
    ],
)
def test_resolve_optimization_split_rejects_invalid_windows(base_request, split_config, message):
    request = PortfolioFactorOptimizationRequest(
        base_request=base_request,
        split=split_config,
    )

    with pytest.raises(ValueError, match=message):
        resolve_optimization_split(request)


def test_calculate_equity_curve_quality_prefers_stable_uptrend_shape():
    smooth_values = [100000 * (1.0005 ** index) for index in range(120)]
    jagged_values = [
        100000 + (index * 700) + (18000 if index % 2 == 0 else -18000)
        for index in range(120)
    ]

    smooth = calculate_equity_curve_quality(
        _summary(final_equity=smooth_values[-1], annual_return_pct=13.0),
        _curve(smooth_values),
    )
    jagged = calculate_equity_curve_quality(
        _summary(
            final_equity=jagged_values[-1],
            annual_return_pct=32.0,
            max_drawdown_pct=28.0,
        ),
        _curve(jagged_values),
    )

    assert smooth.log_equity_trend_r2 > 0.99
    assert smooth.log_equity_trend_r2 > jagged.log_equity_trend_r2
    assert smooth.return_volatility_pct < jagged.return_volatility_pct
    assert smooth.downside_volatility_pct < jagged.downside_volatility_pct
    assert smooth.positive_return_day_ratio > jagged.positive_return_day_ratio


def test_validation_smooth_uptrend_score_penalizes_jagged_high_return_curve():
    train = calculate_equity_curve_quality(
        _summary(final_equity=120000, annual_return_pct=18.0),
        _curve([100000 * (1.0007 ** index) for index in range(120)]),
    )
    smooth_validation = calculate_equity_curve_quality(
        _summary(final_equity=112000, annual_return_pct=12.0),
        _curve([100000 * (1.00045 ** index) for index in range(120)]),
    )
    jagged_validation = calculate_equity_curve_quality(
        _summary(final_equity=142000, annual_return_pct=38.0, max_drawdown_pct=35.0),
        _curve([
            100000 + (index * 900) + (26000 if index % 2 == 0 else -26000)
            for index in range(120)
        ]),
    )

    assert calculate_validation_smooth_uptrend_score(
        train,
        smooth_validation,
    ) > calculate_validation_smooth_uptrend_score(
        train,
        jagged_validation,
    )


def test_build_optimization_risk_flags_marks_volatility_and_low_trend_quality():
    train = calculate_equity_curve_quality(
        _summary(final_equity=170000, annual_return_pct=70.0),
        _curve([100000 * (1.002 ** index) for index in range(120)]),
    )
    validation = calculate_equity_curve_quality(
        _summary(
            final_equity=92000,
            annual_return_pct=-8.0,
            max_drawdown_pct=36.0,
            turnover=12.5,
            rebalances=1,
        ),
        _curve([
            100000 + (index * 120) + (24000 if index % 2 == 0 else -26000)
            for index in range(120)
        ]),
    )

    flags = build_optimization_risk_flags(train, validation)

    assert "negative_validation_return" in flags
    assert "train_validation_gap" in flags
    assert "high_validation_volatility" in flags
    assert "low_equity_trend_quality" in flags
    assert "high_validation_drawdown" in flags
    assert "high_turnover" in flags
    assert "too_few_rebalances" in flags


def test_evaluate_factor_candidate_runs_train_and_validation_with_candidate_config():
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
            top_n=[5],
        ),
    )
    split = resolve_optimization_split(request)
    [candidate] = generate_factor_candidates(request)
    context = PortfolioBacktestContext(
        data_by_symbol={},
        providers={},
        warnings=[],
        diagnostics={},
    )
    calls = []

    def fake_runner(backtest_request, backtest_context, progress_callback=None):
        calls.append(backtest_request)
        assert backtest_context is context
        if backtest_request.metadata["optimization_split"] == "train":
            return _portfolio_result(
                final_equity=118000,
                annual_return_pct=18.0,
                values=[100000 * (1.0007 ** index) for index in range(120)],
                warnings=["shared warning", "train warning"],
            )
        return _portfolio_result(
            final_equity=111000,
            annual_return_pct=11.0,
            values=[100000 * (1.0004 ** index) for index in range(120)],
            warnings=["shared warning", "validation warning"],
        )

    result = evaluate_factor_candidate(
        candidate,
        split,
        context,
        backtest_runner=fake_runner,
    )

    assert [call.metadata["optimization_split"] for call in calls] == [
        "train",
        "validation",
    ]
    assert calls[0].metadata["optimization_candidate_id"] == candidate.candidate_id
    assert calls[1].metadata["optimization_candidate_id"] == candidate.candidate_id
    assert calls[0].factors == candidate.factor_config
    assert calls[1].selection == candidate.selection_config
    assert calls[0].start_date == split.train_start
    assert calls[1].start_date == split.validation_start
    assert result.candidate == candidate
    assert result.train_metrics.annual_return_pct == 18.0
    assert result.validation_metrics.annual_return_pct == 11.0
    assert result.objective_score == calculate_validation_smooth_uptrend_score(
        result.train_metrics,
        result.validation_metrics,
    )
    assert result.warnings == [
        "shared warning",
        "train warning",
        "validation warning",
    ]
    assert "equity_curve" not in result.model_dump(mode="json")
