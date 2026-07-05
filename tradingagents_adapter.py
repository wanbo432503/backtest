import re

from tradingagents_models import ALLOWED_ANALYSTS, TradingAgentsReports


A_SHARE_PREFIX_PATTERN = re.compile(r"^(SH|SZ|BJ)(\d{6})$")
A_SHARE_SUFFIX_PATTERN = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$")
A_SHARE_NUMERIC_PATTERN = re.compile(r"^\d{6}$")


def normalize_a_share_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if not value:
        raise ValueError("symbol must not be empty")

    prefix_match = A_SHARE_PREFIX_PATTERN.match(value)
    if prefix_match:
        return f"{prefix_match.group(1)}{prefix_match.group(2)}"

    suffix_match = A_SHARE_SUFFIX_PATTERN.match(value)
    if suffix_match:
        code, suffix = suffix_match.groups()
        return f"{suffix}{code}"

    if A_SHARE_NUMERIC_PATTERN.match(value):
        if value.startswith(("6", "9")):
            return f"SH{value}"
        if value.startswith(("4", "8")):
            return f"BJ{value}"
        return f"SZ{value}"

    raise ValueError("symbol must be an A-share code")


def to_tradingagents_ticker(symbol: str) -> str:
    normalized = normalize_a_share_symbol(symbol)
    exchange = normalized[:2]
    code = normalized[2:]
    if exchange == "SH":
        return f"{code}.SS"
    if exchange == "SZ":
        return f"{code}.SZ"
    if exchange == "BJ":
        return f"{code}.BJ"
    raise ValueError("unsupported A-share exchange")


def validate_analysts(analysts: list[str]) -> list[str]:
    if not analysts:
        raise ValueError("at least one analyst must be selected")
    unknown = [analyst for analyst in analysts if analyst not in ALLOWED_ANALYSTS]
    if unknown:
        raise ValueError(f"unknown analyst values: {', '.join(unknown)}")
    return analysts


def extract_reports(final_state: dict) -> TradingAgentsReports:
    risk_state = final_state.get("risk_debate_state") or {}
    debate_state = final_state.get("investment_debate_state") or {}
    return TradingAgentsReports(
        market_report=_text_or_none(final_state.get("market_report")),
        sentiment_report=_text_or_none(final_state.get("sentiment_report")),
        news_report=_text_or_none(final_state.get("news_report")),
        fundamentals_report=_text_or_none(final_state.get("fundamentals_report")),
        research_decision=_text_or_none(debate_state.get("judge_decision"))
        or _text_or_none(final_state.get("investment_plan")),
        trader_plan=_text_or_none(final_state.get("trader_investment_plan")),
        risk_discussion=_join_sections(
            [
                ("Aggressive", risk_state.get("aggressive_history")),
                ("Conservative", risk_state.get("conservative_history")),
                ("Neutral", risk_state.get("neutral_history")),
            ]
        ),
        portfolio_decision=_text_or_none(risk_state.get("judge_decision"))
        or _text_or_none(final_state.get("final_trade_decision")),
    )


def _text_or_none(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = "\n".join(str(item) for item in value)
    text = str(value).strip()
    return text or None


def _join_sections(sections: list[tuple[str, object]]) -> str | None:
    parts = []
    for title, content in sections:
        text = _text_or_none(content)
        if text:
            parts.append(f"### {title}\n{text}")
    if not parts:
        return None
    return "\n\n".join(parts)
