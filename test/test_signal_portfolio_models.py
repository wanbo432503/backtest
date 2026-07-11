import pytest
from pydantic import ValidationError

from signal_portfolio_models import SignalPortfolioBacktestRequest


def test_signal_portfolio_accepts_large_manual_a_share_pool():
    symbols = [f"SH60{index:04d}" for index in range(20)]
    request = SignalPortfolioBacktestRequest(
        start_date="2025-01-01",
        end_date="2025-12-31",
        universe={"mode": "manual", "symbols": symbols},
    )

    assert len(request.universe.symbols) == 20
    assert request.risk.max_positions == 10
    assert request.risk.max_position_pct == 0.10
    assert request.risk.target_gross_exposure == 0.85
    assert request.strategy.strategy_name == "trend_pullback_pin_bar"
    assert request.strategy.short_ma_period == 20
    assert request.strategy.medium_ma_period == 60
    assert request.strategy.long_ma_period == 120
    assert request.strategy.ma_distance_pct == 2
    assert request.strategy.volume_multiplier == 1.3
    assert request.strategy.risk_per_trade_pct == 0.5
    assert request.strategy.market_breadth_threshold_pct == 50
    assert request.strategy.cooldown_days == 20


def test_signal_portfolio_rejects_non_a_share_manual_symbol():
    with pytest.raises(ValidationError):
        SignalPortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"mode": "manual", "symbols": ["AAPL"]},
        )


def test_signal_portfolio_requires_strictly_increasing_ma_periods():
    with pytest.raises(ValidationError, match="strictly increasing"):
        SignalPortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"mode": "manual", "symbols": ["SZ002241"]},
            strategy={"short_ma_period": 60, "medium_ma_period": 20},
        )


def test_signal_portfolio_allows_full_market_scan_limit():
    request = SignalPortfolioBacktestRequest(
        start_date="2025-01-01",
        end_date="2025-12-31",
        universe={"mode": "auto", "symbols": [], "max_scan_symbols": 3000},
    )

    assert request.universe.max_scan_symbols == 3000


def test_signal_portfolio_strategy_limits_reward_risk_ratio_to_two_or_three():
    with pytest.raises(ValidationError, match="reward_risk_ratio"):
        SignalPortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"mode": "manual", "symbols": ["SZ002241"]},
            strategy={"reward_risk_ratio": 3.5},
        )

    request = SignalPortfolioBacktestRequest(
        start_date="2025-01-01",
        end_date="2025-12-31",
        universe={"mode": "manual", "symbols": ["SZ002241"]},
    )
    assert request.strategy.reward_risk_ratio == 2.5
    assert request.strategy.min_stop_distance_pct == 1.5
    assert request.strategy.max_stop_distance_pct == 6
    assert "take_profit_pct" not in request.strategy.model_dump()
