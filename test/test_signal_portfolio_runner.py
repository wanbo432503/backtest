import numpy as np
import pandas as pd
import pytest

from signal_portfolio_models import SignalPortfolioBacktestRequest
from signal_portfolio_runner import (
    _build_market_breadth_overlay,
    run_signal_portfolio_with_data,
)
from strategy_library import get_strategy_library


def _frame(periods=280, *, rising=True):
    index = pd.bdate_range("2024-01-02", periods=periods)
    close = np.linspace(10, 20, periods) if rising else np.linspace(20, 10, periods)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(periods, 1_000_000.0),
        },
        index=index,
    )


def _request(strategy_name="rsi_risk_control", parameters=None, **overrides):
    payload = {
        "start_date": "2024-01-02",
        "end_date": "2025-02-28",
        "universe": {
            "mode": "manual",
            "symbols": ["SH603019", "SZ002241"],
        },
        "strategy": {
            "strategy_name": strategy_name,
            "parameters": parameters or {},
        },
        "risk": {"max_positions": 2},
    }
    payload.update(overrides)
    return SignalPortfolioBacktestRequest(**payload)


@pytest.mark.parametrize(
    "strategy_name",
    [definition.strategy_id for definition in get_strategy_library().list()],
)
def test_every_library_strategy_runs_in_signal_portfolio(strategy_name):
    request = _request(strategy_name)

    result = run_signal_portfolio_with_data(
        request,
        {"SH603019": _frame(), "SZ002241": _frame(rising=False)},
    )

    assert result.config["strategy"]["strategy_name"] == strategy_name
    assert result.scan_diagnostics["strategy_id"] == strategy_name
    assert result.summary["final_equity"] > 0
    assert result.equity_curve


def test_market_breadth_is_independent_and_uses_configured_period():
    request = _request(
        market_filter={
            "breadth_ma_period": 5,
            "market_breadth_min_pct": 40,
            "market_breadth_threshold_pct": 60,
            "market_breadth_partial_risk_pct": 50,
        }
    )
    rising = _frame(10, rising=True)
    falling = _frame(10, rising=False)

    overlay, diagnostics = _build_market_breadth_overlay(
        {"SH603019": rising, "SZ002241": falling},
        request.market_filter,
    )

    last_date = rising.index[-1]
    assert overlay.loc[last_date, "breadth_pct"] == 50
    assert overlay.loc[last_date, "risk_multiplier"] == 0.5
    assert diagnostics["breadth_ma_period"] == 5
    assert diagnostics["average_market_breadth_pct"] == 50


def test_disabled_market_filter_always_returns_full_risk():
    request = _request(market_filter={"enabled": False, "breadth_ma_period": 5})

    overlay, diagnostics = _build_market_breadth_overlay(
        {"SH603019": _frame(10)},
        request.market_filter,
    )

    assert set(overlay["risk_multiplier"]) == {1.0}
    assert diagnostics["market_filter_enabled"] is False


def test_signal_portfolio_result_preserves_scan_context_and_normalized_config():
    request = _request("rsi_risk_control", {"rsi_period": 8})

    result = run_signal_portfolio_with_data(
        request,
        {"SH603019": _frame()},
        providers={"SH603019": "mootdx"},
        warnings=["short_history"],
        diagnostics={"requested_symbols": 2},
    )
    payload = result.to_api_response()

    assert payload["data_warnings"] == ["short_history"]
    assert payload["scan_diagnostics"]["requested_symbols"] == 2
    assert payload["scan_diagnostics"]["providers"] == {"SH603019": "mootdx"}
    assert payload["config"]["strategy"]["parameters"]["rsi_period"] == 8
    assert "rsi_buy" in payload["config"]["strategy"]["parameters"]
