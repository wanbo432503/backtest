import pandas as pd
import pytest

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
        trading={"min_commission": 0, "volume_filter": False, "slippage_pct": 0},
        risk={"max_positions": 2, "max_position_pct": 0.5, "target_gross_exposure": 1},
        selection={"top_n": 2, "min_history_bars": 1},
    )


def test_signal_portfolio_uses_shared_cash_and_next_bar_execution(monkeypatch):
    frames = {"SH603019": _frame(100), "SZ002241": _frame(50)}

    def fake_signal_frame(data, request):
        frame = data.copy()
        frame["upper"] = float(frame["Open"].iloc[0]) + 3
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
    assert {trade["reason"] for trade in sells} == {"upper_band"}
    assert result.summary["trades"] == 2
    assert result.summary["final_equity"] > 100000
    assert len(result.symbol_contributions) == 2


def test_signal_portfolio_result_has_diagnostics_and_complete_payload(monkeypatch):
    frames = {"SZ002241": _frame(50)}
    monkeypatch.setattr(
        signal_portfolio_runner,
        "_build_signal_frame",
        lambda data, request: data.assign(
            upper=float(data["Open"].iloc[0]) + 3,
            entry_signal=False,
            signal_strength=0.0,
        ),
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


def test_signal_portfolio_entry_requires_middle_cross_and_two_stable_days():
    before_cross = pd.Series(
        {"Close": 99.0, "middle": 100.0}
    )
    cross_day = pd.Series(
        {"Close": 101.0, "middle": 100.0, "Low": 95.0, "lower": 90.0, "High": 105.0, "upper": 110.0}
    )
    confirmation_day = pd.Series(
        {"Close": 102.0, "middle": 100.5, "Low": 96.0, "lower": 91.0, "High": 106.0, "upper": 111.0}
    )

    assert signal_portfolio_runner._is_two_day_middle_recovery_entry(
        before_cross,
        cross_day,
        confirmation_day,
    )


@pytest.mark.parametrize(
    ("before_updates", "cross_updates", "confirmation_updates"),
    [
        ({"Close": 101.0}, {}, {}),
        ({}, {"Low": 89.0}, {}),
        ({}, {"High": 110.0}, {}),
        ({}, {}, {"Close": 100.0}),
        ({}, {}, {"Low": 90.0}),
        ({}, {}, {"High": 111.0}),
    ],
)
def test_signal_portfolio_entry_rejects_when_recovery_conditions_are_missing(
    before_updates,
    cross_updates,
    confirmation_updates,
):
    before_cross = pd.Series({"Close": 99.0, "middle": 100.0, **before_updates})
    cross_day = pd.Series(
        {
            "Close": 101.0,
            "middle": 100.0,
            "Low": 95.0,
            "lower": 90.0,
            "High": 105.0,
            "upper": 110.0,
            **cross_updates,
        }
    )
    confirmation_day = pd.Series(
        {
            "Close": 102.0,
            "middle": 100.5,
            "Low": 96.0,
            "lower": 91.0,
            "High": 106.0,
            "upper": 111.0,
            **confirmation_updates,
        }
    )

    assert not signal_portfolio_runner._is_two_day_middle_recovery_entry(
        before_cross,
        cross_day,
        confirmation_day,
    )


def test_signal_portfolio_buy_day_upper_touch_exits_next_open(monkeypatch):
    frame = _frame(100)

    def fake_signal_frame(data, request):
        result = data.copy()
        result["upper"] = 100.25
        result["entry_signal"] = False
        result["signal_strength"] = 0.0
        result.loc[result.index[1], "entry_signal"] = True
        result.loc[result.index[1], "signal_strength"] = 0.1
        return result

    monkeypatch.setattr(signal_portfolio_runner, "_build_signal_frame", fake_signal_frame)
    request = _request().model_copy(
        update={"universe": _request().universe.model_copy(update={"symbols": ["SH603019"]})}
    )

    result = run_signal_portfolio_with_data(request, {"SH603019": frame})
    sells = [trade for trade in result.trades if trade["side"] == "sell"]

    assert sells[0]["date"] == "2025-01-06"
    assert sells[0]["price"] == 100.0
    assert sells[0]["reason"] == "upper_band_t1"
