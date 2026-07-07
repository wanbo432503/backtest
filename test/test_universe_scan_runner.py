import pytest

from portfolio_data import PortfolioDataBundle
from portfolio_models import PortfolioBacktestRequest
from stock_universe_provider import StockUniverseRecord, StockUniverseResult
from test.fixtures.portfolio_ohlcv import build_ohlcv_frame
from universe_scan_runner import load_universe_scan_data, run_universe_scan


def _request(**overrides):
    payload = {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "initial_cash": 100000,
        "universe": {"mode": "auto", "symbols": [], "max_scan_symbols": 6},
        "selection": {"top_n": 4, "min_history_bars": 60},
        "rebalance": {"frequency": "monthly", "monthday": 1},
        "trading": {
            "min_commission": 0,
            "volume_filter": False,
            "slippage_pct": 0,
        },
    }
    payload.update(overrides)
    return PortfolioBacktestRequest.model_validate(payload)


def _records():
    return [
        StockUniverseRecord(symbol="SH600000", name="浦发银行", exchange="SH", code_prefix="60", source="fixture"),
        StockUniverseRecord(symbol="SH603019", name="中科曙光", exchange="SH", code_prefix="60", source="fixture"),
        StockUniverseRecord(symbol="SH601318", name="中国平安", exchange="SH", code_prefix="60", source="fixture"),
        StockUniverseRecord(symbol="SZ000001", name="平安银行", exchange="SZ", code_prefix="00", source="fixture"),
        StockUniverseRecord(symbol="SZ002241", name="歌尔股份", exchange="SZ", code_prefix="00", source="fixture"),
        StockUniverseRecord(symbol="SZ000002", name="万科A", exchange="SZ", code_prefix="00", source="fixture"),
    ]


def test_universe_scan_selects_final_holdings_under_five(monkeypatch):
    fixture = {
        "SH600000": build_ohlcv_frame(base_price=10, daily_return=0.0001),
        "SH603019": build_ohlcv_frame(base_price=20, daily_return=0.0012),
        "SH601318": build_ohlcv_frame(base_price=30, daily_return=0.0005),
        "SZ000001": build_ohlcv_frame(base_price=15, daily_return=0.0010),
        "SZ002241": build_ohlcv_frame(base_price=25, daily_return=0.0014),
        "SZ000002": build_ohlcv_frame(base_price=12, daily_return=-0.0003),
    }
    monkeypatch.setattr(
        "universe_scan_runner.get_default_stock_universe",
        lambda **kwargs: StockUniverseResult(records=_records(), source="fixture"),
    )
    monkeypatch.setattr(
        "universe_scan_runner.load_portfolio_ohlcv",
        lambda symbols, *args, **kwargs: PortfolioDataBundle(
            data_by_symbol={symbol: fixture[symbol] for symbol in symbols},
            warnings=[],
            providers={symbol: "fixture" for symbol in symbols},
        ),
    )

    result = run_universe_scan(_request())

    assert 0 < len(result.selected_symbols) <= 4
    assert result.diagnostics["total_universe_size"] == 6
    assert result.diagnostics["loaded_count"] == 6
    assert result.diagnostics["scored_count"] >= len(result.selected_symbols)
    assert result.diagnostics["selected_count"] == len(result.selected_symbols)


def test_universe_scan_handles_partial_data_failures(monkeypatch):
    fixture = {
        "SH600000": build_ohlcv_frame(base_price=10),
        "SZ000001": build_ohlcv_frame(base_price=15),
    }
    monkeypatch.setattr(
        "universe_scan_runner.get_default_stock_universe",
        lambda **kwargs: StockUniverseResult(records=_records()[:4], source="fixture"),
    )
    monkeypatch.setattr(
        "universe_scan_runner.load_portfolio_ohlcv",
        lambda symbols, *args, **kwargs: PortfolioDataBundle(
            data_by_symbol=fixture,
            warnings=["SH603019 获取失败: fixture outage", "SH601318 获取失败: fixture outage"],
            providers={symbol: "fixture" for symbol in fixture},
        ),
    )

    result = run_universe_scan(_request(universe={"mode": "auto", "symbols": [], "max_scan_symbols": 4}, selection={"top_n": 2, "min_history_bars": 60}))

    assert result.selected_symbols
    assert result.diagnostics["load_failed_count"] == 2
    assert result.diagnostics["loaded_count"] == 2
    assert any("fixture outage" in warning for warning in result.warnings)


def test_universe_scan_applies_price_and_volume_prefilters(monkeypatch):
    fixture = {
        "SH600000": build_ohlcv_frame(base_price=2, base_volume=10),
        "SH603019": build_ohlcv_frame(base_price=20, base_volume=1000000),
        "SZ000001": build_ohlcv_frame(base_price=15, base_volume=1000000),
    }
    monkeypatch.setattr(
        "universe_scan_runner.get_default_stock_universe",
        lambda **kwargs: StockUniverseResult(records=[_records()[0], _records()[1], _records()[3]], source="fixture"),
    )
    monkeypatch.setattr(
        "universe_scan_runner.load_portfolio_ohlcv",
        lambda symbols, *args, **kwargs: PortfolioDataBundle(
            data_by_symbol={symbol: fixture[symbol] for symbol in symbols},
            warnings=[],
            providers={symbol: "fixture" for symbol in symbols},
        ),
    )

    result = load_universe_scan_data(
        _request(
            universe={"mode": "auto", "symbols": [], "max_scan_symbols": 3},
            selection={
                "top_n": 2,
                "min_history_bars": 60,
                "min_avg_volume": 100000,
                "min_price": 5,
            },
        )
    )

    assert "SH600000" not in result.data_by_symbol
    assert result.diagnostics["prefilter_skipped_count"] == 1
    assert result.diagnostics["skipped_by_reason"]["below_min_avg_volume"] == 1


def test_manual_diagnostic_mode_still_validates_small_candidate_pool(monkeypatch):
    with pytest.raises(ValueError, match="unsupported_board"):
        load_universe_scan_data(
            _request(
                universe={"mode": "manual", "symbols": ["SH603019", "SZ300750"]},
                selection={"top_n": 1, "min_history_bars": 60},
            )
        )
