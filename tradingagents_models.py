from typing import Any

from pydantic import BaseModel, Field, field_validator

from tradable_universe import TradableUniversePolicy, validate_universe


ALLOWED_TRADINGAGENTS_PROVIDER = "openai_compatible"
ALLOWED_ANALYSTS = {"market", "social", "news", "fundamentals"}
DEFAULT_ANALYSTS = ["market", "news", "fundamentals"]


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


class TradingAgentsConfigView(BaseModel):
    provider: str = ALLOWED_TRADINGAGENTS_PROVIDER
    backend_url: str | None = None
    deep_model: str | None = None
    quick_model: str | None = None
    output_language: str = "Chinese"
    max_debate_rounds: int = 1
    max_risk_rounds: int = 1
    checkpoint_enabled: bool = False
    temperature: float | None = None
    openai_reasoning_effort: str | None = None
    api_key_set: bool = False

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value != ALLOWED_TRADINGAGENTS_PROVIDER:
            raise ValueError("provider must be openai_compatible")
        return value

    @field_validator("backend_url")
    @classmethod
    def validate_backend_url(cls, value: str | None) -> str | None:
        value = _blank_to_none(value)
        if value is None:
            return None
        if not value.startswith(("http://", "https://")):
            raise ValueError("backend_url must start with http:// or https://")
        return value

    @field_validator("max_debate_rounds", "max_risk_rounds")
    @classmethod
    def validate_rounds(cls, value: int) -> int:
        if value < 0 or value > 5:
            raise ValueError("round count must be between 0 and 5")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value < 0 or value > 2:
            raise ValueError("temperature must be between 0 and 2")
        return value


class TradingAgentsConfigUpdate(TradingAgentsConfigView):
    api_key: str | None = None
    clear_api_key: bool = False


class TradingAgentsConfigResponse(BaseModel):
    repo_path: str
    env_path: str
    config: TradingAgentsConfigView


class TradingAgentsConfigTestResponse(BaseModel):
    ok: bool
    checks: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TradingAgentsAnalysisRequest(BaseModel):
    symbol: str
    analysis_date: str
    analysts: list[str] = Field(default_factory=lambda: list(DEFAULT_ANALYSTS))
    max_debate_rounds: int = 1
    max_risk_rounds: int = 1

    @field_validator("symbol", "analysis_date")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("analysts")
    @classmethod
    def validate_analysts(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("analysts must contain at least one analyst")
        unknown = [analyst for analyst in value if analyst not in ALLOWED_ANALYSTS]
        if unknown:
            raise ValueError(f"analysts contains unknown values: {', '.join(unknown)}")
        return value

    @field_validator("max_debate_rounds", "max_risk_rounds")
    @classmethod
    def validate_rounds(cls, value: int) -> int:
        if value < 0 or value > 5:
            raise ValueError("round count must be between 0 and 5")
        return value


class TradingAgentsPortfolioSummaryRequest(BaseModel):
    selected_symbols: list[str]
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    latest_candidate_rankings: list[dict[str, Any]] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    scan_diagnostics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("selected_symbols")
    @classmethod
    def validate_selected_symbols(cls, value: list[str]) -> list[str]:
        result = validate_universe(value, policy=TradableUniversePolicy(max_symbols=4))
        if not result.accepted_symbols:
            raise ValueError("selected_symbols must contain at least one 60/00 stock")
        if not result.ok:
            reasons = [row.reason for row in result.rejected if row.reason]
            reason_text = ", ".join(reasons) if reasons else "invalid_selected_symbols"
            raise ValueError(f"selected_symbols invalid: {reason_text}")
        return result.accepted_symbols


class TradingAgentsReports(BaseModel):
    market_report: str | None = None
    sentiment_report: str | None = None
    news_report: str | None = None
    fundamentals_report: str | None = None
    research_decision: str | None = None
    trader_plan: str | None = None
    risk_discussion: str | None = None
    portfolio_decision: str | None = None


class TradingAgentsAnalysisResponse(BaseModel):
    status: str = "succeeded"
    symbol: str
    tradingagents_ticker: str
    analysis_date: str
    elapsed_seconds: float
    reports: TradingAgentsReports
    warnings: list[str] = Field(default_factory=list)


class TradingAgentsPortfolioSummaryResponse(BaseModel):
    status: str = "succeeded"
    summary_text: str
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)
