import math

import pytest

from portfolio_selection_strategy_library import (
    get_selection_strategy,
    list_selection_strategies,
)


EXPECTED_STRATEGY_IDS = {
    "steady_low_vol_momentum",
    "strong_trend_breakout",
    "high_liquidity_trend",
    "drawdown_control_rotation",
    "value_quality",
}


def test_selection_strategy_library_contains_requested_strategies():
    strategies = list_selection_strategies()
    strategy_ids = {strategy.strategy_id for strategy in strategies}

    assert EXPECTED_STRATEGY_IDS.issubset(strategy_ids)
    assert "custom_factor_blend" in strategy_ids


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


def test_value_quality_strategy_declares_fundamental_and_technical_factors():
    strategy = get_selection_strategy("value_quality")
    factor_keys = {factor.key for factor in strategy.factors}

    assert {"pe_inverse", "pb_inverse", "roe", "profit_growth"}.issubset(factor_keys)
    assert {"ma_trend", "liquidity_turnover"}.issubset(factor_keys)
    assert any("fundamental" in caveat.lower() or "基本面" in caveat for caveat in strategy.caveats)


def test_get_selection_strategy_returns_requested_definition_and_rejects_unknown_id():
    strategy = get_selection_strategy("strong_trend_breakout")

    assert strategy.strategy_id == "strong_trend_breakout"
    assert strategy.name == "强趋势突破策略"

    with pytest.raises(ValueError, match="unknown portfolio selection strategy"):
        get_selection_strategy("missing_strategy")
