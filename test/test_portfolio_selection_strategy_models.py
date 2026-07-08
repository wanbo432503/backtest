import pytest
from pydantic import ValidationError

from portfolio_selection_strategy_models import (
    PortfolioSelectionStrategyConfig,
    PortfolioSelectionStrategyDefinition,
    StrategyFactorSpec,
)


def test_strategy_definition_serializes_to_json():
    definition = PortfolioSelectionStrategyDefinition(
        strategy_id="steady_low_vol_momentum",
        name="稳健低波动动量策略",
        description="Prefer smoother momentum names.",
        suitable_for="Low-turnover portfolio selection.",
        factors=[
            StrategyFactorSpec(
                key="momentum_return",
                label="Momentum return",
                direction="higher_better",
                default_weight=0.4,
                default_lookback=60,
                lookback_candidates=[40, 60, 90],
                weight_candidates=[0.2, 0.4, 0.6],
            ),
            StrategyFactorSpec(
                key="realized_volatility",
                label="Realized volatility",
                direction="lower_better",
                default_weight=-0.3,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[-0.5, -0.3, -0.1],
            ),
        ],
    )

    dumped = definition.model_dump(mode="json")

    assert dumped["strategy_id"] == "steady_low_vol_momentum"
    assert dumped["factors"][0]["direction"] == "higher_better"
    assert dumped["factors"][1]["direction"] == "lower_better"
    assert dumped["default_rebalance_frequency"] == "monthly"


def test_strategy_factor_rejects_invalid_direction_and_non_finite_weights():
    with pytest.raises(ValidationError, match="direction"):
        StrategyFactorSpec(
            key="bad",
            label="Bad",
            direction="bigger",
            default_weight=0.1,
        )

    with pytest.raises(ValidationError, match="finite"):
        StrategyFactorSpec(
            key="bad",
            label="Bad",
            direction="higher_better",
            default_weight=float("nan"),
        )


def test_selection_strategy_config_defaults_to_custom_factor_blend():
    config = PortfolioSelectionStrategyConfig()

    assert config.strategy_id == "custom_factor_blend"
    assert config.enabled is True
    assert config.parameter_overrides == {}


def test_selection_strategy_config_rejects_empty_strategy_id():
    with pytest.raises(ValidationError, match="strategy_id"):
        PortfolioSelectionStrategyConfig(strategy_id="  ")
