import math

import pytest

import portfolio_backtest_runner
from portfolio_fundamentals import FundamentalsBundle
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


def test_portfolio_backtest_uses_named_selection_strategy(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )

    result = run_portfolio_backtest(
        _request(
            selection_strategy={
                "strategy_id": "steady_low_vol_momentum",
                "enabled": True,
            },
        )
    )

    assert result.scan_diagnostics["selection_strategy_id"] == "steady_low_vol_momentum"
    assert result.scan_diagnostics["selection_strategy_name"] == "稳健低波动动量策略"
    assert result.candidate_rankings
    first_ranked = next(row for row in result.candidate_rankings if row["skip_reason"] is None)
    assert first_ranked["strategy_id"] == "steady_low_vol_momentum"
    assert "momentum_return" in first_ranked["strategy_factor_values"]
    assert "normalized_strategy_factors" in first_ranked
    assert result.config["selection_strategy"]["strategy_id"] == "steady_low_vol_momentum"


def test_portfolio_backtest_loads_fundamentals_for_full_financial_strategy(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])
    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        lambda *args, **kwargs: PortfolioDataBundle(data_by_symbol=fixture, warnings=[], providers={}),
    )

    snapshot_dates = []

    def fake_fundamentals(symbols, **kwargs):
        assert symbols == ["SH603019", "SZ002241"]
        snapshot_dates.append(kwargs["as_of_date"])
        return FundamentalsBundle(
            values_by_symbol={
                "SH603019": {
                    "pe_inverse": 0.12,
                    "pb_inverse": 0.4,
                    "roe": 0.2,
                    "gross_margin": 0.4,
                    "operating_cashflow_to_profit": 1.5,
                    "dividend_yield": 0.03,
                },
                "SZ002241": {
                    "pe_inverse": 0.04,
                    "pb_inverse": 0.1,
                    "roe": 0.04,
                    "gross_margin": 0.1,
                    "operating_cashflow_to_profit": 0.3,
                    "dividend_yield": 0.0,
                },
            },
            requested_symbols=["SH603019", "SZ002241"],
            loaded_symbols=["SH603019", "SZ002241"],
            coverage_pct=100.0,
        )

    monkeypatch.setattr("portfolio_backtest_runner.load_portfolio_fundamentals", fake_fundamentals)

    result = run_portfolio_backtest(
        _request(
            selection_strategy={
                "strategy_id": "a_share_full_financial_multifactor",
                "enabled": True,
            },
        )
    )

    assert result.scan_diagnostics["selection_strategy_id"] == "a_share_full_financial_multifactor"
    assert result.scan_diagnostics["loaded_fundamentals"] == 2
    assert snapshot_dates
    assert all(str(date) <= "2025-12-31" for date in snapshot_dates)
    ranked = next(row for row in result.candidate_rankings if row["skip_reason"] is None)
    assert "pe_inverse" in ranked["strategy_factor_values"]


def test_full_financial_strategy_prefilters_large_universe_before_loading_fundamentals(monkeypatch):
    symbols = [f"SH60{index:04d}" for index in range(120)]
    fixture = {
        symbol: build_ohlcv_frame(periods=180, daily_return=0.0005 + index / 1_000_000)
        for index, symbol in enumerate(symbols)
    }
    captured_symbol_batches = []

    def fake_fundamentals(symbols, **kwargs):
        captured_symbol_batches.append(list(symbols))
        return FundamentalsBundle(
            values_by_symbol={
                symbol: {
                    "pe_inverse": 0.1,
                    "pb_inverse": 0.2,
                    "roe": 0.1,
                    "gross_margin": 0.3,
                    "operating_cashflow_to_profit": 1.0,
                    "dividend_yield": 0.02,
                }
                for symbol in symbols
            },
            requested_symbols=list(symbols),
            loaded_symbols=list(symbols),
            coverage_pct=100.0,
        )

    monkeypatch.setattr("portfolio_backtest_runner.load_portfolio_fundamentals", fake_fundamentals)

    context = portfolio_backtest_runner.PortfolioBacktestContext(
        data_by_symbol=fixture,
        providers={symbol: "fixture" for symbol in fixture},
        warnings=[],
        diagnostics={"mode": "manual", "screened_count": len(fixture)},
    )
    request = _request(
        universe={"symbols": symbols[:4]},
        selection={"top_n": 2, "min_history_bars": 60},
        selection_strategy={
            "strategy_id": "a_share_full_financial_multifactor",
            "enabled": True,
        },
        metadata={"fundamental_prefetch_limit": 30},
    )

    result = portfolio_backtest_runner.run_portfolio_backtest_with_context(request, context)

    assert captured_symbol_batches
    assert all(len(batch) <= 30 for batch in captured_symbol_batches)
    assert result.scan_diagnostics["fundamental_prefetch_limit"] == 30
    assert result.scan_diagnostics["fundamental_prefiltered_count"] == 30


def test_default_fundamental_prefetch_limit_is_bounded_for_interactive_backtests():
    assert portfolio_backtest_runner._fundamental_prefetch_limit(
        _request(universe={"mode": "auto"}, selection={"top_n": 20, "min_history_bars": 60}),
        3179,
    ) == 20
    assert portfolio_backtest_runner._fundamental_prefetch_limit(
        _request(selection={"top_n": 2, "min_history_bars": 60}),
        3179,
    ) == 10
    assert portfolio_backtest_runner._fundamental_prefetch_limit(
        _request(universe={"mode": "auto"}, selection={"top_n": 20, "min_history_bars": 60}),
        30,
    ) == 20


def test_portfolio_backtest_context_loads_once_and_can_be_reused(monkeypatch):
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])
    loader_calls = []

    def fake_loader(symbols, *args, **kwargs):
        loader_calls.append(list(symbols))
        return PortfolioDataBundle(
            data_by_symbol=fixture,
            warnings=[],
            providers={symbol: "fixture" for symbol in fixture},
        )

    monkeypatch.setattr("portfolio_backtest_runner.load_portfolio_ohlcv", fake_loader)
    request = _request()
    alternate_request = _request(
        factors={
            "momentum_weight": 0.1,
            "volatility_weight": -0.1,
            "liquidity_weight": 0.6,
            "trend_weight": 0.2,
        }
    )

    context = portfolio_backtest_runner.load_portfolio_backtest_context(request)
    first = portfolio_backtest_runner.run_portfolio_backtest_with_context(request, context)
    second = portfolio_backtest_runner.run_portfolio_backtest_with_context(alternate_request, context)

    assert len(loader_calls) == 1
    assert context.providers == {symbol: "fixture" for symbol in fixture}
    assert context.diagnostics["mode"] == "manual"
    assert first.to_api_response().keys() == run_portfolio_backtest(request).to_api_response().keys()
    assert second.equity_curve


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
