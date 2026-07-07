import pytest
from pydantic import ValidationError

from optimization_models import AShareTradingConfig
from portfolio_models import (
    PortfolioBacktestRequest,
    PortfolioBacktestResult,
    UniverseConfig,
)


def test_portfolio_backtest_request_defaults_are_prototype_ready():
    request = PortfolioBacktestRequest(start_date="2025-01-01", end_date="2025-12-31")

    assert request.universe.symbols == ["SH603019", "SZ002241"]
    assert request.universe.max_symbols == 4
    assert request.universe.allowed_code_prefixes == ("60", "00")
    assert request.selection.top_n == 2
    assert request.risk.max_position_pct == 0.50
    assert request.risk.target_gross_exposure == 0.95
    assert isinstance(request.trading, AShareTradingConfig)


def test_portfolio_backtest_request_rejects_invalid_date_order():
    with pytest.raises(ValidationError, match="start_date must be earlier than end_date"):
        PortfolioBacktestRequest(start_date="2025-12-31", end_date="2025-01-01")


def test_portfolio_backtest_request_rejects_more_than_four_symbols():
    with pytest.raises(ValidationError, match="too_many_symbols"):
        PortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={
                "symbols": ["SH600000", "SH601318", "SH603019", "SH605001", "SZ000001"],
            },
        )


def test_portfolio_backtest_request_rejects_top_n_larger_than_symbol_count():
    with pytest.raises(ValidationError, match="top_n must not exceed symbol count"):
        PortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"symbols": ["SH603019", "SZ002241"]},
            selection={"top_n": 3},
        )


def test_portfolio_backtest_request_rejects_non_60_00_symbols_before_data_loading():
    with pytest.raises(ValidationError, match="unsupported_board"):
        PortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"symbols": ["SH603019", "SZ300750"]},
        )


def test_universe_config_dedupes_duplicate_symbols():
    config = UniverseConfig(symbols=["SH603019", "603019", "SZ002241"])

    assert config.symbols == ["SH603019", "SZ002241"]


def test_portfolio_backtest_result_to_api_response_contains_required_keys():
    result = PortfolioBacktestResult(
        summary={"final_equity": 101000.0},
        equity_curve=[],
        positions=[],
        trades=[],
        rebalance_events=[],
        candidate_rankings=[],
        data_warnings=[],
        risk_flags=[],
        config={"selection": {"top_n": 1}},
    )

    response = result.to_api_response()

    assert set(response) == {
        "summary",
        "equity_curve",
        "positions",
        "trades",
        "rebalance_events",
        "candidate_rankings",
        "data_warnings",
        "risk_flags",
        "config",
    }
    assert response["trades"] == []
    assert response["data_warnings"] == []
