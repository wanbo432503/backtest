import pytest
from pydantic import ValidationError

from signal_portfolio_models import SignalPortfolioBacktestRequest
from strategy_library import get_strategy_library


def _request(**overrides):
    payload = {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "universe": {"mode": "manual", "symbols": ["SZ002241"]},
    }
    payload.update(overrides)
    return SignalPortfolioBacktestRequest(**payload)


def test_signal_portfolio_uses_generic_nested_strategy_and_separate_market_filter():
    request = _request(
        strategy={
            "strategy_name": "rsi_risk_control",
            "parameters": {"rsi_period": 9},
        },
        market_filter={"breadth_ma_period": 80},
    )

    assert request.strategy.strategy_name == "rsi_risk_control"
    assert request.strategy.parameters == {"rsi_period": 9}
    assert request.market_filter.breadth_ma_period == 80
    assert request.market_filter.market_breadth_min_pct == 40


def test_signal_portfolio_normalization_validates_and_completes_strategy_defaults():
    request = _request(
        strategy={
            "strategy_name": "rsi_risk_control",
            "parameters": {"rsi_period": 9},
        }
    )

    normalized = request.normalized_for_library(get_strategy_library())

    assert normalized.strategy.parameters["rsi_period"] == 9
    assert "rsi_buy" in normalized.strategy.parameters
    assert normalized.selection.min_history_bars >= 9
    assert request.strategy.parameters == {"rsi_period": 9}


def test_signal_portfolio_rejects_unknown_strategy_parameter_during_normalization():
    request = _request(
        strategy={
            "strategy_name": "rsi_risk_control",
            "parameters": {"not_a_parameter": 1},
        }
    )

    with pytest.raises(ValueError, match="not_a_parameter"):
        request.normalized_for_library(get_strategy_library())


def test_signal_portfolio_accepts_legacy_flat_pin_bar_configuration():
    request = _request(
        strategy={
            "short_ma_period": 15,
            "market_breadth_min_pct": 35,
            "market_breadth_threshold_pct": 55,
            "market_breadth_partial_risk_pct": 60,
        }
    )

    assert request.strategy.strategy_name == "trend_pullback_pin_bar"
    assert request.strategy.parameters["short_ma_period"] == 15
    assert request.market_filter.market_breadth_min_pct == 35
    assert request.market_filter.market_breadth_threshold_pct == 55
    assert request.market_filter.market_breadth_partial_risk_pct == 60


def test_signal_portfolio_rejects_ambiguous_legacy_and_nested_market_filter():
    with pytest.raises(ValidationError, match="market_filter"):
        _request(
            strategy={"market_breadth_min_pct": 35},
            market_filter={"market_breadth_min_pct": 30},
        )


def test_signal_portfolio_requires_breadth_minimum_below_full_risk_threshold():
    with pytest.raises(ValidationError, match="market_breadth_min_pct"):
        _request(
            market_filter={
                "market_breadth_min_pct": 50,
                "market_breadth_threshold_pct": 50,
            }
        )


def test_signal_portfolio_rejects_non_a_share_manual_symbol():
    with pytest.raises(ValidationError):
        _request(universe={"mode": "manual", "symbols": ["AAPL"]})


def test_signal_portfolio_auto_mode_allows_full_market_scan_limit():
    request = _request(
        universe={"mode": "auto", "symbols": [], "max_scan_symbols": 3000}
    )

    assert request.universe.max_scan_symbols == 3000
