import pytest
import pandas as pd

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


def test_load_portfolio_fundamentals_extracts_akshare_valuation_quality_cashflow_snapshot():
    def akshare_loader(symbol: str) -> dict[str, pd.DataFrame]:
        assert symbol == "SH600000"
        return {
            "valuation": pd.DataFrame(
                [
                    {
                        "数据日期": "2025-06-27",
                        "PE(TTM)": 10,
                        "市净率": 2,
                        "市销率": 4,
                        "市现率": 5,
                        "总市值": 1000,
                        "当日收盘价": 10,
                    }
                ]
            ),
            "profit": pd.DataFrame(
                [
                    {
                        "NOTICE_DATE": "2025-04-20",
                        "REPORT_DATE": "2025-03-31",
                        "OPERATE_INCOME": 200,
                        "OPERATE_COST": 120,
                        "PARENT_NETPROFIT": 30,
                        "NETPROFIT": 35,
                    }
                ]
            ),
            "balance": pd.DataFrame(
                [
                    {
                        "NOTICE_DATE": "2025-04-20",
                        "REPORT_DATE": "2025-03-31",
                        "ASSET_BALANCE": 500,
                        "LIAB_BALANCE": 200,
                        "PARENT_EQUITY_BALANCE": 250,
                    }
                ]
            ),
            "cash_flow": pd.DataFrame(
                [
                    {
                        "NOTICE_DATE": "2025-04-20",
                        "REPORT_DATE": "2025-03-31",
                        "NETCASH_OPERATE": 45,
                        "CONSTRUCT_LONG_ASSET": 10,
                        "ASSIGN_DIVIDEND_PORFIT": 15,
                    }
                ]
            ),
            "dividend": pd.DataFrame(
                [
                    {
                        "公告日期": "2025-05-15",
                        "除权除息日": "2025-05-22",
                        "派息": 2.0,
                        "进度": "实施",
                    },
                    {
                        "公告日期": "2023-05-15",
                        "除权除息日": "2023-05-22",
                        "派息": 1.0,
                        "进度": "实施",
                    },
                ]
            ),
        }

    bundle = load_portfolio_fundamentals(
        ["SH600000"],
        data_provider="akshare",
        as_of_date="2025-06-30",
        akshare_loader=akshare_loader,
    )

    values = bundle.values_by_symbol["SH600000"]
    assert values["pe_inverse"] == pytest.approx(0.1)
    assert values["pb_inverse"] == pytest.approx(0.5)
    assert values["ps_inverse"] == pytest.approx(0.25)
    assert values["pcf_inverse"] == pytest.approx(0.2)
    assert values["roe"] == pytest.approx(30 / 250)
    assert values["roa"] == pytest.approx(35 / 500)
    assert values["gross_margin"] == pytest.approx(0.4)
    assert values["net_margin"] == pytest.approx(0.15)
    assert values["debt_to_assets"] == pytest.approx(0.4)
    assert values["operating_cashflow_to_profit"] == pytest.approx(1.5)
    assert values["fcf_yield"] == pytest.approx(0.035)
    assert values["dividend_yield"] == pytest.approx(0.02)
    assert values["dividend_stability"] == pytest.approx(0.4)
    assert values["dividend_coverage"] == pytest.approx(3.0)
