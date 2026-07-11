import pytest
from pathlib import Path

import tradingagents_adapter
from tradingagents_adapter import (
    TradingAgentsAdapterError,
    build_portfolio_summary_prompt,
    build_run_config,
    extract_reports,
    load_tradingagents_env,
    normalize_a_share_symbol,
    run_tradingagents_portfolio_summary,
    run_tradingagents_analysis,
    sanitize_error_message,
    temporary_environ,
    temporary_sys_path,
    to_tradingagents_ticker,
    validate_analysts,
)
from tradingagents_models import TradingAgentsAnalysisRequest, TradingAgentsPortfolioSummaryRequest


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


def test_sanitize_error_message_masks_secret_values():
    message = "provider failed with abc123xyz"

    sanitized = sanitize_error_message(message, {"OPENAI_COMPATIBLE_API_KEY": "abc123xyz"})

    assert "abc123xyz" not in sanitized
    assert "<redacted>" in sanitized


def test_temporary_sys_path_restores_original_value(tmp_path):
    import sys

    original = list(sys.path)
    with temporary_sys_path(tmp_path):
        assert str(tmp_path) == sys.path[0]

    assert sys.path == original


def test_temporary_environ_restores_original_value(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "old")

    with temporary_environ({"TRADINGAGENTS_LLM_PROVIDER": "new"}):
        import os

        assert os.environ["TRADINGAGENTS_LLM_PROVIDER"] == "new"

    import os

    assert os.environ["TRADINGAGENTS_LLM_PROVIDER"] == "old"


def test_load_tradingagents_env_reads_values(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("TRADINGAGENTS_LLM_PROVIDER=openai_compatible\n", encoding="utf-8")

    values = load_tradingagents_env(env_path)

    assert values["TRADINGAGENTS_LLM_PROVIDER"] == "openai_compatible"


def test_build_run_config_merges_env_and_request_rounds(tmp_path):
    env_values = {
        "TRADINGAGENTS_LLM_PROVIDER": "openai_compatible",
        "TRADINGAGENTS_LLM_BACKEND_URL": "http://localhost:1234/v1",
        "TRADINGAGENTS_DEEP_THINK_LLM": "deep",
        "TRADINGAGENTS_QUICK_THINK_LLM": "quick",
        "TRADINGAGENTS_OUTPUT_LANGUAGE": "Chinese",
        "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "1",
        "TRADINGAGENTS_MAX_RISK_ROUNDS": "1",
    }
    request = TradingAgentsAnalysisRequest(
        symbol="SZ002241",
        analysis_date="2026-07-05",
        max_debate_rounds=2,
        max_risk_rounds=3,
    )

    config = build_run_config(request, env_values, project_path=tmp_path)

    assert config["llm_provider"] == "openai_compatible"
    assert config["backend_url"] == "http://localhost:1234/v1"
    assert config["deep_think_llm"] == "deep"
    assert config["quick_think_llm"] == "quick"
    assert config["output_language"] == "Chinese"
    assert config["max_debate_rounds"] == 2
    assert config["max_risk_discuss_rounds"] == 3


def test_run_tradingagents_analysis_stays_in_current_process_when_legacy_python_env_is_set(
    monkeypatch, tmp_path
):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TRADINGAGENTS_LLM_PROVIDER=openai_compatible\n"
        "TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1\n"
        "TRADINGAGENTS_DEEP_THINK_LLM=deep\n"
        "TRADINGAGENTS_QUICK_THINK_LLM=quick\n",
        encoding="utf-8",
    )
    legacy_python = tmp_path / "legacy-python"
    legacy_python.write_text("#!/usr/bin/env sh\nexit 99\n", encoding="utf-8")
    legacy_python.chmod(0o755)
    monkeypatch.setenv("TRADINGAGENTS_PYTHON", str(legacy_python))
    captured = {}

    class FakeGraph:
        def __init__(self, selected_analysts, config, debug=False):
            captured["selected_analysts"] = selected_analysts
            captured["config"] = config

        def propagate(self, company_name, trade_date, asset_type="stock"):
            captured["company_name"] = company_name
            return {"market_report": "market"}

    monkeypatch.setattr(tradingagents_adapter, "_load_default_config", lambda: {})
    monkeypatch.setattr(tradingagents_adapter, "_load_tradingagents_graph_class", lambda: FakeGraph)

    response = run_tradingagents_analysis(
        TradingAgentsAnalysisRequest(symbol="SZ002241", analysis_date="2026-07-05"),
        env_path=env_path,
        project_path=tmp_path,
    )

    assert captured["selected_analysts"] == ["market", "news", "fundamentals"]
    assert captured["company_name"] == "002241.SZ"
    assert response.reports.market_report == "market"


def test_run_tradingagents_analysis_uses_fake_graph(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "TRADINGAGENTS_LLM_PROVIDER=openai_compatible",
                "TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1",
                "TRADINGAGENTS_DEEP_THINK_LLM=deep",
                "TRADINGAGENTS_QUICK_THINK_LLM=quick",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeGraph:
        def __init__(self, selected_analysts, config, debug=False):
            captured["selected_analysts"] = selected_analysts
            captured["config"] = config
            captured["debug"] = debug

        def propagate(self, company_name, trade_date, asset_type="stock"):
            captured["company_name"] = company_name
            captured["trade_date"] = trade_date
            captured["asset_type"] = asset_type
            return {
                "market_report": "market",
                "trader_investment_plan": "trader",
                "risk_debate_state": {"judge_decision": "portfolio"},
            }

    response = run_tradingagents_analysis(
        TradingAgentsAnalysisRequest(
            symbol="SZ002241",
            analysis_date="2026-07-05",
            analysts=["market", "news"],
        ),
        env_path=env_path,
        project_path=tmp_path,
        graph_class=FakeGraph,
    )

    assert captured["selected_analysts"] == ["market", "news"]
    assert captured["company_name"] == "002241.SZ"
    assert captured["trade_date"] == "2026-07-05"
    assert response.symbol == "SZ002241"
    assert response.tradingagents_ticker == "002241.SZ"
    assert response.reports.market_report == "market"
    assert response.reports.trader_plan == "trader"
    assert response.reports.portfolio_decision == "portfolio"
    assert response.elapsed_seconds >= 0


def test_build_portfolio_summary_prompt_is_deterministic_and_explanation_only():
    request = TradingAgentsPortfolioSummaryRequest(
        selected_symbols=["SH603019", "SZ002241"],
        summary_metrics={"final_equity": 101000, "max_drawdown_pct": -8.2},
        latest_candidate_rankings=[
            {"symbol": "SH603019", "score": 0.91, "momentum": 0.2},
            {"symbol": "SZ002241", "score": 0.73, "momentum": 0.1},
        ],
        risk_flags=["high_drawdown"],
    )

    prompt = build_portfolio_summary_prompt(request)

    assert "SH603019" in prompt
    assert "SZ002241" in prompt
    assert "high_drawdown" in prompt
    assert "解释" in prompt
    assert "不要改写" in prompt


def test_run_portfolio_summary_uses_completion_client_and_masks_secret(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1\n"
        "TRADINGAGENTS_DEEP_THINK_LLM=deep\n"
        "OPENAI_COMPATIBLE_API_KEY=secret-token\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_client(prompt, env_values):
        captured["prompt"] = prompt
        captured["api_key"] = env_values.get("OPENAI_COMPATIBLE_API_KEY")
        return "组合解释文本"

    response = run_tradingagents_portfolio_summary(
        TradingAgentsPortfolioSummaryRequest(
            selected_symbols=["SH603019"],
            summary_metrics={"final_equity": 101000},
            latest_candidate_rankings=[],
            risk_flags=[],
        ),
        env_path=env_path,
        completion_client=fake_client,
    )

    assert captured["api_key"] == "secret-token"
    assert "secret-token" not in response.model_dump_json()
    assert response.summary_text == "组合解释文本"
    assert response.warnings == ["AI summary is explanatory only and is not used by backtest metrics."]


def test_run_portfolio_summary_sanitizes_client_errors(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_COMPATIBLE_API_KEY=secret-token\n", encoding="utf-8")

    def broken_client(prompt, env_values):
        raise RuntimeError("backend leaked secret-token")

    with pytest.raises(TradingAgentsAdapterError, match="<redacted>"):
        run_tradingagents_portfolio_summary(
            TradingAgentsPortfolioSummaryRequest(
                selected_symbols=["SH603019"],
                summary_metrics={},
                latest_candidate_rankings=[],
                risk_flags=[],
            ),
            env_path=env_path,
            completion_client=broken_client,
        )


def test_run_tradingagents_analysis_accepts_tradingagents_tuple_result(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "TRADINGAGENTS_LLM_PROVIDER=openai_compatible",
                "TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1",
                "TRADINGAGENTS_DEEP_THINK_LLM=deep",
                "TRADINGAGENTS_QUICK_THINK_LLM=quick",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeGraph:
        def __init__(self, selected_analysts, config, debug=False):
            pass

        def propagate(self, company_name, trade_date, asset_type="stock"):
            return (
                {
                    "market_report": "market",
                    "news_report": "news",
                    "final_trade_decision": "buy",
                    "risk_debate_state": {"judge_decision": "portfolio"},
                },
                "processed signal",
            )

    response = run_tradingagents_analysis(
        TradingAgentsAnalysisRequest(symbol="SZ002241", analysis_date="2026-07-05"),
        env_path=env_path,
        project_path=tmp_path,
        graph_class=FakeGraph,
    )

    assert response.reports.market_report == "market"
    assert response.reports.news_report == "news"
    assert response.reports.portfolio_decision == "portfolio"


def test_run_tradingagents_analysis_sanitizes_graph_errors(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1\n"
        "OPENAI_COMPATIBLE_API_KEY=dummy\n",
        encoding="utf-8",
    )

    class BrokenGraph:
        def __init__(self, selected_analysts, config, debug=False):
            raise RuntimeError("failed with dummy")

    with pytest.raises(TradingAgentsAdapterError) as exc_info:
        run_tradingagents_analysis(
            TradingAgentsAnalysisRequest(symbol="SZ002241", analysis_date="2026-07-05"),
            env_path=env_path,
            project_path=tmp_path,
            graph_class=BrokenGraph,
        )

    assert "dummy" not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)


def test_run_tradingagents_analysis_wraps_import_errors(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1\n", encoding="utf-8")

    with pytest.raises(TradingAgentsAdapterError):
        run_tradingagents_analysis(
            TradingAgentsAnalysisRequest(symbol="SZ002241", analysis_date="2026-07-05"),
            env_path=env_path,
            project_path=tmp_path,
        )
