import pytest

from portfolio_fundamentals import (
    FundamentalsBundle,
    load_portfolio_fundamentals,
)


def test_load_portfolio_fundamentals_extracts_value_quality_factors_with_injected_loader():
    loaded_symbols = []

    def info_loader(yfinance_symbol: str) -> dict:
        loaded_symbols.append(yfinance_symbol)
        return {
            "trailingPE": 10,
            "priceToBook": 2,
            "returnOnEquity": 0.16,
            "revenueGrowth": 0.08,
            "earningsGrowth": 0.12,
        }

    bundle = load_portfolio_fundamentals(["SH600000"], info_loader=info_loader)

    assert isinstance(bundle, FundamentalsBundle)
    assert loaded_symbols == ["600000.SS"]
    assert bundle.values_by_symbol["SH600000"]["pe_inverse"] == pytest.approx(0.1)
    assert bundle.values_by_symbol["SH600000"]["pb_inverse"] == pytest.approx(0.5)
    assert bundle.values_by_symbol["SH600000"]["roe"] == pytest.approx(0.16)
    assert bundle.values_by_symbol["SH600000"]["revenue_growth"] == pytest.approx(0.08)
    assert bundle.values_by_symbol["SH600000"]["profit_growth"] == pytest.approx(0.12)
    assert bundle.coverage_pct == pytest.approx(100.0)
    assert bundle.missing_symbols == []
    assert bundle.warnings == []


def test_load_portfolio_fundamentals_reports_missing_and_low_coverage_without_crashing():
    def info_loader(yfinance_symbol: str) -> dict:
        if yfinance_symbol == "600000.SS":
            return {"trailingPE": 12}
        raise RuntimeError("provider unavailable")

    bundle = load_portfolio_fundamentals(
        ["SH600000", "SZ002241", "SH603019"],
        info_loader=info_loader,
        min_coverage_pct=50,
    )

    diagnostics = bundle.to_diagnostics()

    assert bundle.values_by_symbol["SH600000"]["pe_inverse"] == pytest.approx(1 / 12)
    assert "SZ002241" in bundle.missing_symbols
    assert "SH603019" in bundle.missing_symbols
    assert bundle.coverage_pct == pytest.approx(33.3333333333)
    assert any("fundamental coverage low" in warning for warning in bundle.warnings)
    assert diagnostics["requested_symbols"] == 3
    assert diagnostics["loaded_fundamentals"] == 1
    assert diagnostics["missing_symbols"] == ["SZ002241", "SH603019"]


def test_fundamentals_bundle_defaults_are_safe_for_empty_symbol_lists():
    bundle = load_portfolio_fundamentals([], info_loader=lambda symbol: {})

    assert bundle.values_by_symbol == {}
    assert bundle.coverage_pct == 0.0
    assert bundle.to_diagnostics()["requested_symbols"] == 0
