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
            "High": [price + 0.5, price + 0.5, price + 0.5, price + 6, price + 0.5, price + 0.5, price + 0.5, price + 0.5],
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
        price = float(frame["Open"].iloc[0])
        frame["ma_short"] = price - 1
        frame["ma_medium"] = price - 2
        frame["atr"] = price * 0.02
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
    assert {trade["reason"] for trade in sells} == {"reward_target"}
    assert {trade["symbol"]: trade["shares"] for trade in buys} == {
        "SH603019": 200,
        "SZ002241": 400,
    }
    assert result.summary["trades"] == 2
    assert result.summary["final_equity"] > 100000
    assert len(result.symbol_contributions) == 2


def test_signal_portfolio_result_has_diagnostics_and_complete_payload(monkeypatch):
    frames = {"SZ002241": _frame(50)}
    monkeypatch.setattr(
        signal_portfolio_runner,
        "_build_signal_frame",
        lambda data, request: data.assign(
            ma_short=float(data["Open"].iloc[0]) - 1,
            ma_medium=float(data["Open"].iloc[0]) - 2,
            atr=float(data["Open"].iloc[0]) * 0.02,
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
    assert payload["scan_diagnostics"]["market_breadth_threshold_pct"] == 50
    assert payload["scan_diagnostics"]["average_market_breadth_pct"] == 100


def test_market_breadth_filter_requires_more_than_half_above_medium_ma():
    date = pd.Timestamp("2025-01-02")

    def breadth_frame(close):
        return pd.DataFrame(
            {"Close": [close], "ma_medium": [100.0], "entry_signal": [True]},
            index=[date],
        )

    passing_frames = {
        "A": breadth_frame(101),
        "B": breadth_frame(102),
        "C": breadth_frame(99),
    }
    passing = signal_portfolio_runner._apply_market_breadth_filter(
        passing_frames,
        _request(),
    )

    assert all(bool(frame.loc[date, "entry_signal"]) for frame in passing_frames.values())
    assert passing["breadth_blocked_signal_count"] == 0

    blocked_frames = {
        "A": breadth_frame(101),
        "B": breadth_frame(99),
        "C": breadth_frame(98),
    }
    blocked = signal_portfolio_runner._apply_market_breadth_filter(
        blocked_frames,
        _request(),
    )

    assert not any(bool(frame.loc[date, "entry_signal"]) for frame in blocked_frames.values())
    assert blocked["breadth_blocked_signal_count"] == 3

    exactly_half_frames = {"A": breadth_frame(101), "B": breadth_frame(99)}
    signal_portfolio_runner._apply_market_breadth_filter(exactly_half_frames, _request())

    assert not any(bool(frame.loc[date, "entry_signal"]) for frame in exactly_half_frames.values())


def _pin_bar_row(**updates):
    values = {
        "Open": 100.0,
        "High": 103.0,
        "Low": 90.0,
        "Close": 101.0,
        "Volume": 130.0,
        "ma_short": 100.0,
        "ma_medium": 95.0,
        "ma_long": 90.0,
        "support": 90.0,
        "average_volume": 100.0,
        "atr": 3.0,
    }
    return pd.Series({**values, **updates})


def test_signal_portfolio_entry_requires_trend_pullback_pin_bar_and_volume():
    config = _request().strategy

    assert signal_portfolio_runner._is_trend_pullback_pin_bar(_pin_bar_row(), config)


@pytest.mark.parametrize(
    "updates",
    [
        {"ma_short": 94.0},
        {"Close": 110.0, "Low": 100.0, "support": 90.0},
        {"Low": 99.0},
        {"High": 110.0},
        {"Volume": 129.0},
    ],
)
def test_signal_portfolio_entry_rejects_when_any_core_filter_is_missing(updates):
    config = _request().strategy

    assert not signal_portfolio_runner._is_trend_pullback_pin_bar(
        _pin_bar_row(**updates),
        config,
    )


def test_signal_portfolio_buy_day_reward_target_exits_next_open(monkeypatch):
    frame = _frame(100)
    frame.loc[frame.index[2], "High"] = 106.0

    def fake_signal_frame(data, request):
        result = data.copy()
        result["ma_short"] = 99.0
        result["ma_medium"] = 98.0
        result["atr"] = 2.0
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
    assert sells[0]["reason"] == "reward_target_t1"


@pytest.mark.parametrize("entry_day_open", [100.0, 104.0])
def test_signal_portfolio_skips_missing_breakout_or_excessive_gap(monkeypatch, entry_day_open):
    frame = _frame(100)
    frame.loc[frame.index[2], "Open"] = entry_day_open
    frame.loc[frame.index[2], "High"] = 100.5 if entry_day_open == 100.0 else 105.0

    def fake_signal_frame(data, request):
        result = data.copy()
        result["ma_short"] = 99.0
        result["ma_medium"] = 98.0
        result["atr"] = 2.0
        result["entry_signal"] = False
        result["signal_strength"] = 0.0
        result.loc[result.index[1], "High"] = 101.0 if entry_day_open == 100.0 else 100.5
        result.loc[result.index[1], "entry_signal"] = True
        result.loc[result.index[1], "signal_strength"] = 0.1
        return result

    monkeypatch.setattr(signal_portfolio_runner, "_build_signal_frame", fake_signal_frame)
    request = _request().model_copy(
        update={"universe": _request().universe.model_copy(update={"symbols": ["SH603019"]})}
    )

    result = run_signal_portfolio_with_data(request, {"SH603019": frame})

    assert not [trade for trade in result.trades if trade["side"] == "buy"]


def test_signal_portfolio_trend_weakness_exits_at_next_open(monkeypatch):
    frame = _frame(100)
    frame.loc[frame.index[3], ["High", "Close"]] = [100.5, 98.0]
    frame.loc[frame.index[4], ["High", "Close"]] = [100.5, 98.0]

    def fake_signal_frame(data, request):
        result = data.copy()
        result["ma_short"] = 99.0
        result["ma_medium"] = 98.0
        result["atr"] = 2.0
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

    assert sells[0]["date"] == "2025-01-08"
    assert sells[0]["price"] == 100.0
    assert sells[0]["reason"] == "trend_weak"


def test_signal_portfolio_cooldown_blocks_repeat_signal_after_exit(monkeypatch):
    frame = _frame(100)

    def fake_signal_frame(data, request):
        result = data.copy()
        result["ma_short"] = 99.0
        result["ma_medium"] = 98.0
        result["atr"] = 2.0
        result["entry_signal"] = False
        result["signal_strength"] = 0.0
        result.loc[result.index[[1, 4]], "entry_signal"] = True
        result.loc[result.index[[1, 4]], "signal_strength"] = 0.1
        return result

    monkeypatch.setattr(signal_portfolio_runner, "_build_signal_frame", fake_signal_frame)
    request = _request().model_copy(
        update={"universe": _request().universe.model_copy(update={"symbols": ["SH603019"]})}
    )

    result = run_signal_portfolio_with_data(request, {"SH603019": frame})
    buys = [trade for trade in result.trades if trade["side"] == "buy"]

    assert len(buys) == 1
    assert len(result.signal_events) == 1
