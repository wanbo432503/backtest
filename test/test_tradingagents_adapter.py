import pytest

from tradingagents_adapter import (
    extract_reports,
    normalize_a_share_symbol,
    to_tradingagents_ticker,
    validate_analysts,
)


@pytest.mark.parametrize(
    ("raw", "normalized", "ticker"),
    [
        ("SH603019", "SH603019", "603019.SS"),
        ("603019.SH", "SH603019", "603019.SS"),
        ("SZ002241", "SZ002241", "002241.SZ"),
        ("002241.SZ", "SZ002241", "002241.SZ"),
        ("BJ430047", "BJ430047", "430047.BJ"),
        ("430047.BJ", "BJ430047", "430047.BJ"),
        ("600519", "SH600519", "600519.SS"),
        ("002241", "SZ002241", "002241.SZ"),
    ],
)
def test_a_share_symbol_conversion(raw, normalized, ticker):
    assert normalize_a_share_symbol(raw) == normalized
    assert to_tradingagents_ticker(raw) == ticker


@pytest.mark.parametrize("raw", ["AAPL", "0700.HK", "BTC-USD", "", "12345"])
def test_non_a_share_symbols_are_rejected(raw):
    with pytest.raises(ValueError):
        normalize_a_share_symbol(raw)

    with pytest.raises(ValueError):
        to_tradingagents_ticker(raw)


def test_validate_analysts_accepts_known_values_in_input_order():
    assert validate_analysts(["news", "market"]) == ["news", "market"]


def test_validate_analysts_rejects_empty_or_unknown_values():
    with pytest.raises(ValueError, match="at least one"):
        validate_analysts([])

    with pytest.raises(ValueError, match="unknown"):
        validate_analysts(["market", "invalid"])


def test_extract_reports_maps_final_state_sections():
    reports = extract_reports(
        {
            "market_report": "market",
            "sentiment_report": "sentiment",
            "news_report": "news",
            "fundamentals_report": "fundamentals",
            "investment_debate_state": {
                "bull_history": "bull",
                "bear_history": "bear",
                "judge_decision": "research decision",
            },
            "trader_investment_plan": "trader",
            "risk_debate_state": {
                "aggressive_history": "aggressive",
                "conservative_history": "conservative",
                "neutral_history": "neutral",
                "judge_decision": "portfolio decision",
            },
        }
    )

    assert reports.market_report == "market"
    assert reports.sentiment_report == "sentiment"
    assert reports.news_report == "news"
    assert reports.fundamentals_report == "fundamentals"
    assert reports.research_decision == "research decision"
    assert reports.trader_plan == "trader"
    assert "aggressive" in reports.risk_discussion
    assert "conservative" in reports.risk_discussion
    assert "neutral" in reports.risk_discussion
    assert reports.portfolio_decision == "portfolio decision"


def test_extract_reports_handles_missing_sections():
    reports = extract_reports({})

    assert reports.market_report is None
    assert reports.risk_discussion is None
    assert reports.portfolio_decision is None
