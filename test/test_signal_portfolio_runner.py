import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from pydantic import BaseModel, ConfigDict

import signal_portfolio_runner
from market_data import DataSourceResult
from portfolio_data import load_portfolio_ohlcv
from signal_portfolio_models import (
    SignalPortfolioBacktestRequest,
    SignalPortfolioBacktestResult,
)
from signal_portfolio_runner import (
    _build_market_breadth_overlay,
    run_signal_portfolio_with_data,
)
from strategy_library import get_strategy_library
from strategy_library import StrategyLibrary
from strategy_engine import EntryIntent, StrategyDecision, StrategyDefinition
from strategies.volume_divergence_rsi_long import STRATEGY_DEFINITION as LONG_DEFINITION


class _PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


def _prepared_long_trade_frame(data, config):
    frame = data.copy()
    frame["ma_value"] = 100.0
    frame["average_volume"] = 1_000_000.0
    frame["volume_confirmed"] = False
    frame["macd_dif"] = -1.0
    frame["bottom_divergence"] = False
    frame["bottom_divergence_recent"] = False
    frame["rsi"] = 50.0
    previous, signal, fill, stop = frame.index[59:63]
    frame.loc[previous, ["Close", "ma_value", "rsi"]] = [99, 100, 29]
    frame.loc[signal, ["Open", "High", "Low", "Close", "ma_value", "Volume", "volume_confirmed", "bottom_divergence_recent", "rsi"]] = [
        100, 102, 99, 101, 100, 1_300_000, True, True, 31,
    ]
    frame.loc[fill, ["Open", "High", "Low", "Close"]] = [99, 106, 98.5, 105]
    frame.loc[stop, ["Open", "High", "Low", "Close"]] = [104, 105, 103, 104]
    return frame


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


def test_new_strategy_portfolio_path_fills_ten_percent_and_trailing_stop():
    definition = replace(
        LONG_DEFINITION,
        prepare_frame=_prepared_long_trade_frame,
    )
    library = StrategyLibrary([definition])
    request = _request(
        definition.strategy_id,
        market_filter={"enabled": False},
        initial_cash=100_000,
    )

    result = run_signal_portfolio_with_data(
        request,
        {"SH603019": _frame()},
        strategy_library=library,
    )

    buys = [trade for trade in result.trades if trade["side"] == "buy"]
    sells = [trade for trade in result.trades if trade["side"] == "sell"]
    assert buys[0]["shares"] == 100
    assert buys[0]["amount"] <= 10_000
    assert sells[0]["reason"] == "stop_loss"
    assert sells[0]["price"] == pytest.approx(103.88 * (1 - 0.0005))


def test_loaded_signal_portfolio_executes_at_raw_price(monkeypatch):
    definition = StrategyDefinition(
        strategy_id="dual_price_pipeline",
        display_name="Dual Price Pipeline",
        description="integration fixture",
        config_model=_PipelineConfig,
        parameters=(),
        prepare_frame=lambda data, config: data.copy(),
        evaluate=lambda context: StrategyDecision(
            entry=EntryIntent("next_open")
        )
        if context.bar_index == 0 and context.position is None
        else StrategyDecision(),
        min_history_bars=lambda config: 1,
    )
    library = StrategyLibrary([definition])
    dual_frame = pd.DataFrame(
        {
            "Open": [6.6, 6.6, 6.6],
            "High": [6.7, 6.7, 6.7],
            "Low": [6.5, 6.5, 6.5],
            "Close": [6.6, 6.6, 6.6],
            "Volume": [1_000, 1_000, 1_000],
            "RawOpen": [10.0, 10.0, 6.6],
            "RawHigh": [10.1, 10.1, 6.7],
            "RawLow": [9.9, 9.9, 6.5],
            "RawClose": [10.0, 10.0, 6.6],
            "AdjFactor": [0.66, 0.66, 1.0],
            "CashDividendPer10": [0.0, 0.0, 1.0],
            "BonusSharesPer10": [0.0, 0.0, 5.0],
            "RightsSharesPer10": [0.0, 0.0, 0.0],
            "RightsPrice": [0.0, 0.0, 0.0],
        },
        index=pd.bdate_range("2024-01-02", periods=3),
    )
    dual_frame.attrs["corporate_actions"] = [
        {
            "date": "2024-01-04",
            "CashDividendPer10": 1.0,
            "BonusSharesPer10": 5.0,
            "RightsSharesPer10": 0.0,
            "RightsPrice": 0.0,
            "RawReferencePrice": 6.6,
        }
    ]
    monkeypatch.setattr(
        "portfolio_data.fetch_ohlcv",
        lambda *args, **kwargs: DataSourceResult(dual_frame, "mootdx", []),
    )
    bundle = load_portfolio_ohlcv(
        ["SH603019"],
        "2024-01-02",
        "2024-01-04",
        min_history_bars=1,
    )
    request = _request(
        definition.strategy_id,
        initial_cash=1_000,
        market_filter={"enabled": False},
        risk={
            "max_positions": 1,
            "max_position_pct": 1,
            "target_gross_exposure": 1,
            "max_drawdown_stop_pct": None,
        },
        trading={
            "t_plus_one": True,
            "lot_size": 1,
            "limit_up_down_filter": False,
            "volume_filter": False,
            "slippage_pct": 0,
            "buy_commission_pct": 0,
            "sell_commission_pct": 0,
            "stamp_tax_pct": 0,
            "min_commission": 0,
        },
    )

    result = run_signal_portfolio_with_data(
        request,
        bundle.data_by_symbol,
        strategy_library=library,
    )

    buy = result.trades[0]
    assert buy["price"] == 10
    assert buy["shares"] == 100
    assert result.positions[0]["shares"] == 150
    assert result.scan_diagnostics["corporate_action_events"][0]["cash_dividend"] == 10
    assert result.summary["final_equity"] == 1_000


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


def test_ma60_portfolio_preloads_history_before_backtest_start():
    definition = get_strategy_library().get("ma60_price_cross")
    other_definition = get_strategy_library().get("rsi_risk_control")

    assert hasattr(signal_portfolio_runner, "_portfolio_data_start_date")
    assert (
        signal_portfolio_runner._portfolio_data_start_date("2025-01-01", definition)
        == "2023-07-01"
    )
    assert (
        signal_portfolio_runner._portfolio_data_start_date(
            "2025-01-01", other_definition
        )
        == "2025-01-01"
    )


def test_ma60_portfolio_observes_first_year_before_trading():
    frame = _frame()
    request = _request(
        "ma60_price_cross",
        {"max_entry_gap_pct": 20},
        market_filter={"enabled": False},
    )

    result = run_signal_portfolio_with_data(
        request,
        {"SH603019": frame},
    )

    buys = [trade for trade in result.trades if trade["side"] == "buy"]
    assert buys
    assert pd.Timestamp(buys[0]["date"]) >= frame.index[250]
    assert result.scan_diagnostics["entry_observation_start_date"] == "2024-01-02"
    assert result.scan_diagnostics["entry_observation_bars_required"] == 250
    assert result.scan_diagnostics["indicator_warmup_bars"] == 60
    assert result.scan_diagnostics["insufficient_entry_history_count"] > 0


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


def test_signal_portfolio_api_bounds_event_details_and_removes_strategy_states():
    response_limit = 500
    total = response_limit + 3
    result = SignalPortfolioBacktestResult(
        summary={"final_equity": 100_000},
        signal_events=[
            {"date": f"2026-01-{index:04d}", "symbol": "SH603019"}
            for index in range(total)
        ],
        scan_diagnostics={
            "signal_count": total,
            "strategy_states": {"SH603019": {"armed": True}},
        },
    )

    payload = result.to_api_response()

    assert len(payload["signal_events"]) == response_limit
    assert payload["signal_events"][0]["date"] == "2026-01-0003"
    assert payload["scan_diagnostics"]["signal_count"] == total
    assert payload["scan_diagnostics"]["signal_events_returned"] == response_limit
    assert payload["scan_diagnostics"]["signal_events_truncated"] is True
    assert "strategy_states" not in payload["scan_diagnostics"]


def test_signal_portfolio_keeps_exact_breadth_counts_when_events_are_capped():
    definition = StrategyDefinition(
        strategy_id="many_blocked_signals",
        display_name="Many Blocked Signals",
        description="capacity regression fixture",
        config_model=_PipelineConfig,
        parameters=(),
        prepare_frame=lambda data, config: data.copy(),
        evaluate=lambda context: StrategyDecision(entry=EntryIntent("next_open"))
        if context.position is None
        else StrategyDecision(),
        min_history_bars=lambda config: 1,
    )
    frame = _frame(periods=520, rising=False)
    request = _request(
        definition.strategy_id,
        start_date="2024-01-02",
        end_date="2026-12-31",
        initial_cash=1_000,
        market_filter={"enabled": True, "breadth_ma_period": 2},
    )

    result = run_signal_portfolio_with_data(
        request,
        {"SH603019": frame},
        strategy_library=StrategyLibrary([definition]),
    )

    assert result.scan_diagnostics["signal_count"] == 520
    assert result.scan_diagnostics["breadth_blocked_signal_count"] == 520
    assert result.scan_diagnostics["signal_events_returned"] == 500
    assert result.scan_diagnostics["signal_events_truncated"] is True
    assert len(result.signal_events) == 500
