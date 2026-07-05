import pytest
from pydantic import ValidationError

from tradingagents_models import (
    TradingAgentsAnalysisRequest,
    TradingAgentsConfigUpdate,
    TradingAgentsConfigView,
    TradingAgentsReports,
)


def test_config_view_defaults_to_openai_compatible_chinese_output():
    config = TradingAgentsConfigView()

    assert config.provider == "openai_compatible"
    assert config.output_language == "Chinese"
    assert config.max_debate_rounds == 1
    assert config.max_risk_rounds == 1
    assert config.api_key_set is False


def test_config_update_rejects_non_openai_compatible_provider():
    with pytest.raises(ValidationError, match="provider"):
        TradingAgentsConfigUpdate(provider="openai")


def test_config_update_rejects_backend_url_without_http_scheme():
    with pytest.raises(ValidationError, match="backend_url"):
        TradingAgentsConfigUpdate(backend_url="localhost:1234/v1")


def test_config_update_rejects_rounds_outside_allowed_range():
    with pytest.raises(ValidationError, match="max_debate_rounds"):
        TradingAgentsConfigUpdate(max_debate_rounds=6)

    with pytest.raises(ValidationError, match="max_risk_rounds"):
        TradingAgentsConfigUpdate(max_risk_rounds=-1)


def test_config_update_rejects_temperature_outside_allowed_range():
    with pytest.raises(ValidationError, match="temperature"):
        TradingAgentsConfigUpdate(temperature=2.1)


def test_analysis_request_requires_at_least_one_known_analyst():
    with pytest.raises(ValidationError, match="analysts"):
        TradingAgentsAnalysisRequest(symbol="SZ002241", analysis_date="2026-07-05", analysts=[])

    with pytest.raises(ValidationError, match="analysts"):
        TradingAgentsAnalysisRequest(
            symbol="SZ002241",
            analysis_date="2026-07-05",
            analysts=["market", "unknown"],
        )


def test_analysis_request_defaults_to_core_a_share_analysts():
    request = TradingAgentsAnalysisRequest(symbol="SZ002241", analysis_date="2026-07-05")

    assert request.analysts == ["market", "news", "fundamentals"]
    assert request.max_debate_rounds == 1
    assert request.max_risk_rounds == 1


def test_reports_default_to_none_for_missing_sections():
    reports = TradingAgentsReports()

    assert reports.market_report is None
    assert reports.portfolio_decision is None
