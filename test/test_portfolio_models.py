import pytest
from pydantic import ValidationError

from optimization_models import AShareTradingConfig
from portfolio_models import (
    PortfolioBacktestRequest,
    PortfolioBacktestResult,
    SelectionConfig,
    UniverseConfig,
)


def test_portfolio_backtest_request_defaults_are_prototype_ready():
    request = PortfolioBacktestRequest(start_date="2025-01-01", end_date="2025-12-31")

    assert request.universe.mode == "auto"
    assert request.universe.symbols == []
    assert request.universe.max_symbols == 4
    assert request.universe.ohlcv_batch_size == 20
    assert request.universe.ohlcv_batch_delay_seconds == 0.0
    assert request.universe.ohlcv_request_delay_seconds == 0.0
    assert request.universe.allowed_code_prefixes == ("60", "00")
    assert request.selection.top_n == 2
    assert request.risk.max_position_pct == 0.50
    assert request.risk.target_gross_exposure == 0.95
    assert isinstance(request.trading, AShareTradingConfig)


def test_selection_config_allows_phase31_top_n_up_to_twenty():
    assert SelectionConfig(top_n=20).top_n == 20

    with pytest.raises(ValidationError, match="top_n"):
        SelectionConfig(top_n=21)


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


def test_portfolio_backtest_request_rejects_invalid_rate_limit_values():
    with pytest.raises(ValidationError, match="ohlcv_batch_size must be greater than 0"):
        PortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"mode": "auto", "ohlcv_batch_size": 0},
        )

    with pytest.raises(ValidationError, match="rate limit delays must be non-negative"):
        PortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"mode": "auto", "ohlcv_request_delay_seconds": -0.1},
        )


def test_portfolio_backtest_request_rejects_top_n_larger_than_symbol_count():
    with pytest.raises(ValidationError, match="top_n must not exceed symbol count"):
        PortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"mode": "manual", "symbols": ["SH603019", "SZ002241"]},
            selection={"top_n": 3},
        )


def test_portfolio_backtest_request_rejects_non_60_00_symbols_before_data_loading():
    with pytest.raises(ValidationError, match="unsupported_board"):
        PortfolioBacktestRequest(
            start_date="2025-01-01",
            end_date="2025-12-31",
            universe={"mode": "manual", "symbols": ["SH603019", "SZ300750"]},
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
        "scan_diagnostics",
        "config",
    }
    assert response["trades"] == []
    assert response["data_warnings"] == []
