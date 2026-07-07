import math

import pytest

from portfolio_backtest_runner import run_portfolio_backtest
from portfolio_data import PortfolioDataBundle
from portfolio_models import PortfolioBacktestRequest
from test.fixtures.portfolio_ohlcv import build_ohlcv_frame, build_portfolio_ohlcv_fixture


def _request(**overrides):
    payload = {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "initial_cash": 100000,
        "universe": {"symbols": ["SH603019", "SZ002241"]},
        "selection": {"top_n": 1, "min_history_bars": 60},
        "rebalance": {"frequency": "monthly", "monthday": 1},
        "trading": {
            "min_commission": 0,
            "volume_filter": False,
            "slippage_pct": 0,
        },
        "risk": {
            "max_position_pct": 0.50,
            "target_gross_exposure": 0.95,
            "cash_buffer_pct": 0.05,
        },
    }
    payload.update(overrides)
    return PortfolioBacktestRequest.model_validate(payload)


def test_portfolio_backtest_produces_complete_result(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )

    result = run_portfolio_backtest(_request()).to_api_response()

    assert result["equity_curve"]
    assert result["rebalance_events"]
    assert result["candidate_rankings"]
    assert result["trades"]
    assert result["positions"]
    assert set(result) == {
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
    assert result["scan_diagnostics"]["mode"] == "manual"
    assert math.isfinite(result["summary"]["final_equity"])
    assert math.isfinite(result["summary"]["sharpe"])


def test_portfolio_backtest_selection_changes_after_momentum_changes(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )

    result = run_portfolio_backtest(_request())
    selected_sets = [tuple(event["selected_symbols"]) for event in result.rebalance_events]

    assert ("SH603019",) in selected_sets
    assert ("SZ002241",) in selected_sets


def test_portfolio_backtest_buys_lot_rounded_shares_and_reduces_cash(monkeypatch):
    fixture = {"SH603019": build_ohlcv_frame(base_price=20), "SZ002241": build_ohlcv_frame(base_price=25)}
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )

    result = run_portfolio_backtest(_request(universe={"symbols": ["SH603019", "SZ002241"]}))
    first_buy = next(trade for trade in result.trades if trade["side"] == "buy")
    first_buy_curve = next(point for point in result.equity_curve if point["date"] >= first_buy["date"])

    assert first_buy["shares"] % 100 == 0
    assert first_buy_curve["cash"] < 100000


def test_portfolio_backtest_respects_max_single_position_cap(monkeypatch):
    fixture = {"SH603019": build_ohlcv_frame(base_price=10), "SZ002241": build_ohlcv_frame(base_price=25)}
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )

    result = run_portfolio_backtest(_request(risk={"max_position_pct": 0.30, "target_gross_exposure": 0.95}))
    buy = next(trade for trade in result.trades if trade["side"] == "buy")

    assert buy["amount"] <= 100000 * 0.30


def test_portfolio_backtest_sells_before_buys_on_rebalance(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )

    result = run_portfolio_backtest(_request())

    for event in result.rebalance_events:
        trade_sides = [trade["side"] for trade in event["trades"]]
        if "sell" in trade_sides and "buy" in trade_sides:
            assert trade_sides.index("sell") < trade_sides.index("buy")


def test_portfolio_backtest_skips_limit_up_buy(monkeypatch):
    limit_up = build_ohlcv_frame(base_price=20, limit_up_days=[65])
    other = build_ohlcv_frame(base_price=25, daily_return=-0.001)
    fixture = {"SH603019": limit_up, "SZ002241": other}
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )
    request = _request(
        start_date=str(limit_up.index[0].date()),
        end_date=str(limit_up.index[90].date()),
        rebalance={"frequency": "weekly", "weekday": int(limit_up.index[65].weekday())},
    )

    result = run_portfolio_backtest(request)

    assert any(skip["reason"] == "limit_up" for event in result.rebalance_events for skip in event["skipped_trades"])


def test_portfolio_backtest_skips_limit_down_sell(monkeypatch):
    held = build_ohlcv_frame(base_price=20, limit_down_days=[45])
    replacement = build_ohlcv_frame(base_price=30)
    fixture = {"SH603019": held, "SZ002241": replacement}

    def fake_scores(data_by_symbol, as_of_date, *args):
        selected = "SH603019" if as_of_date < held.index[45] else "SZ002241"
        other = "SZ002241" if selected == "SH603019" else "SH603019"
        return [
            {"symbol": selected, "score": 1.0, "rank": 1, "skip_reason": None, "factor_values": {}},
            {"symbol": other, "score": 0.0, "rank": 2, "skip_reason": None, "factor_values": {}},
        ]

    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )
    monkeypatch.setattr("portfolio_backtest_runner.score_candidates", fake_scores)

    request = _request(
        start_date=str(held.index[0].date()),
        end_date=str(held.index[70].date()),
        selection={"top_n": 1, "min_history_bars": 10},
        rebalance={"frequency": "weekly", "weekday": int(held.index[45].weekday())},
    )
    result = run_portfolio_backtest(request)

    assert any(skip["reason"] == "limit_down" for event in result.rebalance_events for skip in event["skipped_trades"])


def test_portfolio_backtest_does_not_sell_same_day_buys(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )

    result = run_portfolio_backtest(_request())

    buy_keys = {(trade["date"], trade["symbol"]) for trade in result.trades if trade["side"] == "buy"}
    sell_keys = {(trade["date"], trade["symbol"]) for trade in result.trades if trade["side"] == "sell"}
    assert buy_keys.isdisjoint(sell_keys)


def test_portfolio_backtest_flags_underinvested_result(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )
    monkeypatch.setattr(
        "portfolio_backtest_runner.score_candidates",
        lambda *args, **kwargs: [
            {"symbol": "SH603019", "score": None, "rank": None, "skip_reason": "insufficient_history", "factor_values": {}},
            {"symbol": "SZ002241", "score": None, "rank": None, "skip_reason": "insufficient_history", "factor_values": {}},
        ],
    )

    result = run_portfolio_backtest(_request(selection={"top_n": 1, "min_history_bars": 60}))

    assert "underinvested" in result.risk_flags


def test_portfolio_backtest_raises_for_invalid_universe_before_data_loading(monkeypatch):
    called = False

    def fake_loader(*args, **kwargs):
        nonlocal called
        called = True
        return PortfolioDataBundle()

    monkeypatch.setattr("portfolio_backtest_runner.load_portfolio_ohlcv", fake_loader)

    with pytest.raises(ValueError, match="unsupported_board"):
        run_portfolio_backtest(
            PortfolioBacktestRequest.model_construct(
                start_date="2025-01-01",
                end_date="2025-12-31",
                initial_cash=100000,
                data_provider="auto",
                universe=type("Universe", (), {"symbols": ["SZ300750"], "max_symbols": 4})(),
                selection=type("Selection", (), {"top_n": 1, "min_history_bars": 60})(),
                rebalance=type("Rebalance", (), {"frequency": "monthly", "monthday": 1, "weekday": 0})(),
                trading=PortfolioBacktestRequest(start_date="2025-01-01", end_date="2025-12-31").trading,
                risk=PortfolioBacktestRequest(start_date="2025-01-01", end_date="2025-12-31").risk,
                metadata={},
            )
        )

    assert called is False
