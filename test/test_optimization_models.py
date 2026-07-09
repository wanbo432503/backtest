import pytest
from pydantic import ValidationError

from optimization_models import (
    AShareTradingConfig,
    OptimizationConfig,
    OptimizationRequest,
    RiskConfig,
    StrategyParamConfig,
)


def test_risk_config_defaults_are_safe_for_fixed_a_share_backtests():
    config = RiskConfig()

    assert config.enabled is True
    assert config.position_pct == 0.95
    assert config.stop_loss_pct == 5
    assert config.take_profit_pct == 12
    assert config.max_account_drawdown_pct == 30


def test_a_share_trading_config_defaults_match_phase_2_rules():
    config = AShareTradingConfig()

    assert config.long_only is True
    assert config.t_plus_one is True
    assert config.lot_size == 100
    assert config.limit_up_down_filter is True
    assert config.stamp_tax_pct == 0.05
    assert config.min_commission == 5


def test_optimization_config_defaults_focus_on_one_a_share_target():
    config = OptimizationConfig()

    assert config.enabled is False
    assert config.symbols == ["SH603019"]
    assert config.objective == "score"
    assert config.top_n == 10
    assert config.max_combinations == 300
    assert config.max_workers == 8
    assert config.min_trades == 5


def test_optimization_config_rejects_multiple_symbols():
    with pytest.raises(ValidationError):
        OptimizationConfig(symbols=["SH603019", "SZ002241"])


def test_strategy_param_config_keeps_fixed_params_and_search_space():
    config = StrategyParamConfig(
        strategy_name="rsi_risk_control",
        fixed_params={"trend_ma": 60},
        search_space={"rsi_period": [6, 14], "rsi_buy": [25, 30]},
    )

    assert config.fixed_params["trend_ma"] == 60
    assert config.search_space["rsi_buy"] == [25, 30]


def test_optimization_request_groups_all_phase_2_configs():
    request = OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        strategies=[StrategyParamConfig(strategy_name="rsi_risk_control")],
    )

    assert request.risk_config.position_pct == 0.95
    assert request.a_share_config.lot_size == 100
    assert request.optimization_config.strategies[0].strategy_name == "rsi_risk_control"


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_pct", 0),
        ("position_pct", 1.2),
    ],
)
def test_risk_config_rejects_invalid_position_pct(field, value):
    with pytest.raises(ValidationError):
        RiskConfig(**{field: value})


def test_a_share_trading_config_rejects_invalid_lot_size():
    with pytest.raises(ValidationError):
        AShareTradingConfig(lot_size=0)


def test_optimization_config_allows_user_defined_combination_count_above_1000():
    config = OptimizationConfig(max_combinations=5000)

    assert config.max_combinations == 5000


def test_optimization_config_rejects_non_positive_combination_count():
    with pytest.raises(ValidationError):
        OptimizationConfig(max_combinations=0)


def test_optimization_config_rejects_invalid_max_workers():
    with pytest.raises(ValidationError):
        OptimizationConfig(max_workers=0)

    with pytest.raises(ValidationError):
        OptimizationConfig(max_workers=9)
