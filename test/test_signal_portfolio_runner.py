import pandas as pd

import signal_portfolio_runner
from signal_portfolio_models import SignalPortfolioBacktestRequest
from signal_portfolio_runner import run_signal_portfolio_with_data


def _frame(price=100.0):
    index = pd.date_range("2025-01-01", periods=8, freq="B")
    return pd.DataFrame(
        {
            "Open": [price] * 8,
            "High": [price + 0.5, price + 0.5, price + 0.5, price + 5, price + 0.5, price + 0.5, price + 0.5, price + 0.5],
            "Low": [price - 0.2] * 8,
            "Close": [price] * 8,
            "Volume": [1_000_000] * 8,
        },
        index=index,
    )


def _request():
    return SignalPortfolioBacktestRequest(
        start_date="2025-01-01",
        end_date="2025-01-31",
        initial_cash=100000,
        universe={"mode": "manual", "symbols": ["SH603019", "SZ002241"]},
        strategy={"stop_loss_pct": 1, "take_profit_pct": 3},
        trading={"min_commission": 0, "volume_filter": False, "slippage_pct": 0},
        risk={"max_positions": 2, "max_position_pct": 0.5, "target_gross_exposure": 1},
        selection={"top_n": 2, "min_history_bars": 1},
    )


def test_signal_portfolio_uses_shared_cash_and_next_bar_execution(monkeypatch):
    frames = {"SH603019": _frame(100), "SZ002241": _frame(50)}

    def fake_signal_frame(data, request):
        frame = data.copy()
        frame["entry_signal"] = False
        frame["signal_strength"] = 0.0
        frame.loc[frame.index[1], "entry_signal"] = True
        frame.loc[frame.index[1], "signal_strength"] = 0.1
        return frame

    monkeypatch.setattr(signal_portfolio_runner, "_build_signal_frame", fake_signal_frame)

    result = run_signal_portfolio_with_data(_request(), frames)
    buys = [trade for trade in result.trades if trade["side"] == "buy"]
    sells = [trade for trade in result.trades if trade["side"] == "sell"]

    assert len(buys) == 2
    assert {trade["date"] for trade in buys} == {"2025-01-03"}
    assert len(sells) == 2
    assert {trade["reason"] for trade in sells} == {"take_profit"}
    assert result.summary["trades"] == 2
    assert result.summary["final_equity"] > 100000
    assert len(result.symbol_contributions) == 2


def test_signal_portfolio_result_has_diagnostics_and_complete_payload(monkeypatch):
    frames = {"SZ002241": _frame(50)}
    monkeypatch.setattr(
        signal_portfolio_runner,
        "_build_signal_frame",
        lambda data, request: data.assign(entry_signal=False, signal_strength=0.0),
    )
    request = _request().model_copy(
        update={"universe": _request().universe.model_copy(update={"symbols": ["SZ002241"]})}
    )

    payload = run_signal_portfolio_with_data(request, frames).to_api_response()

    assert set(payload) == {
        "summary",
        "equity_curve",
        "positions",
        "trades",
        "symbol_contributions",
        "signal_events",
        "data_warnings",
        "risk_flags",
        "scan_diagnostics",
        "config",
    }
    assert payload["scan_diagnostics"]["loaded_symbols"] == 1
    assert payload["scan_diagnostics"]["signal_count"] == 0
