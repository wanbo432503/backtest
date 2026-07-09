import math

import pytest

from portfolio_selection_strategy_library import (
    build_factor_search_space_for_strategy,
    get_selection_strategy,
    list_selection_strategies,
)


EXPECTED_STRATEGY_IDS = {
    "steady_low_vol_momentum",
    "strong_trend_breakout",
    "high_liquidity_trend",
    "drawdown_control_rotation",
    "a_share_full_financial_multifactor",
}

REMOVED_STRATEGY_IDS = {"value_quality", "custom_factor_blend"}


def test_selection_strategy_library_contains_requested_strategies():
    strategies = list_selection_strategies()
    strategy_ids = {strategy.strategy_id for strategy in strategies}

    assert EXPECTED_STRATEGY_IDS.issubset(strategy_ids)
    assert REMOVED_STRATEGY_IDS.isdisjoint(strategy_ids)


def test_selection_strategy_ids_are_unique_and_deterministic():
    first = list_selection_strategies()
    second = list_selection_strategies()

    assert [strategy.strategy_id for strategy in first] == [strategy.strategy_id for strategy in second]
    assert len({strategy.strategy_id for strategy in first}) == len(first)


def test_each_requested_strategy_has_complete_factor_definitions():
    strategies = {
        strategy.strategy_id: strategy
        for strategy in list_selection_strategies()
        if strategy.strategy_id in EXPECTED_STRATEGY_IDS
    }

    for strategy_id, strategy in strategies.items():
        assert strategy.name
        assert strategy.description
        assert strategy.suitable_for
        assert len(strategy.factors) >= 3, strategy_id
        assert 1 <= strategy.default_top_n <= 20
        assert strategy.top_n_candidates
        assert max(strategy.top_n_candidates) <= 20
        for factor in strategy.factors:
            assert factor.key
            assert factor.label
            assert factor.direction in {"higher_better", "lower_better"}
            assert math.isfinite(factor.default_weight)
            assert factor.lookback_candidates or factor.weight_candidates


def test_get_selection_strategy_returns_requested_definition_and_rejects_unknown_id():
    strategy = get_selection_strategy("strong_trend_breakout")

    assert strategy.strategy_id == "strong_trend_breakout"
    assert strategy.name == "强趋势突破策略"

    with pytest.raises(ValueError, match="unknown portfolio selection strategy"):
        get_selection_strategy("missing_strategy")

    for removed_strategy_id in REMOVED_STRATEGY_IDS:
        with pytest.raises(ValueError, match="unknown portfolio selection strategy"):
            get_selection_strategy(removed_strategy_id)


def test_each_strategy_can_generate_valid_optimization_search_space():
    for strategy_id in EXPECTED_STRATEGY_IDS:
        search_space = build_factor_search_space_for_strategy(strategy_id)

        assert search_space.strategy_id == strategy_id
        assert search_space.factor_weights
        assert search_space.top_n
        assert 1 <= min(search_space.top_n)
        assert max(search_space.top_n) <= 20
        assert search_space.score_threshold == [None]
        for candidates in search_space.factor_weights.values():
            assert candidates
            assert all(math.isfinite(candidate) for candidate in candidates)
        for candidates in search_space.factor_lookbacks.values():
            assert candidates
            assert all(candidate > 0 for candidate in candidates)


def test_strategy_search_space_exposes_legacy_factor_space_for_compatible_factors():
    search_space = build_factor_search_space_for_strategy("steady_low_vol_momentum")

    assert search_space.legacy_factor_search_space is not None
    assert search_space.legacy_factor_search_space.momentum_lookback == [40, 60, 90, 120]
    assert search_space.legacy_factor_search_space.volatility_lookback == [10, 20, 40]
    assert search_space.legacy_factor_search_space.liquidity_lookback == [10, 20, 40]
    assert search_space.legacy_factor_search_space.top_n == [2, 3, 5, 10]


def test_full_financial_strategy_exposes_required_fundamental_and_risk_factors():
    strategy = get_selection_strategy("a_share_full_financial_multifactor")
    factor_keys = {factor.key for factor in strategy.factors}

    assert strategy.name == "A股完整财务多因子策略"
    assert strategy.default_rebalance_frequency == "monthly"
    assert strategy.default_top_n == 5
    assert {
        "pe_inverse",
        "pb_inverse",
        "ps_inverse",
        "roe",
        "gross_margin",
        "debt_to_assets",
        "operating_cashflow_to_profit",
        "fcf_yield",
        "dividend_yield",
        "dividend_stability",
        "realized_volatility",
        "max_drawdown_window",
        "seasoned_momentum",
        "recent_overheat_return",
    }.issubset(factor_keys)
