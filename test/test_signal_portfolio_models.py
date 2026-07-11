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
    assert request.risk.max_positions == 5
    assert request.strategy.strategy_name == "boll_macd_breakout"


def test_signal_portfolio_rejects_non_a_share_manual_symbol():
    with pytest.raises(ValidationError):
        SignalPortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"mode": "manual", "symbols": ["AAPL"]},
        )


def test_signal_portfolio_requires_slow_macd_period_above_fast_period():
    with pytest.raises(ValidationError, match="slow_period"):
        SignalPortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"mode": "manual", "symbols": ["SZ002241"]},
            strategy={"fast_period": 26, "slow_period": 12},
        )


def test_signal_portfolio_allows_full_market_scan_limit():
    request = SignalPortfolioBacktestRequest(
        start_date="2025-01-01",
        end_date="2025-12-31",
        universe={"mode": "auto", "symbols": [], "max_scan_symbols": 3000},
    )

    assert request.universe.max_scan_symbols == 3000
