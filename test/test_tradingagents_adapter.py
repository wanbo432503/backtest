import pytest
import json
from pathlib import Path

import tradingagents_adapter
from tradingagents_adapter import (
    TradingAgentsAdapterError,
    build_run_config,
    extract_reports,
    find_tradingagents_python,
    load_tradingagents_env,
    normalize_a_share_symbol,
    run_tradingagents_analysis,
    sanitize_error_message,
    temporary_environ,
    temporary_sys_path,
    to_tradingagents_ticker,
    validate_analysts,
)
from tradingagents_models import TradingAgentsAnalysisRequest


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

    config = build_run_config(request, env_values, repo_path=tmp_path)

    assert config["llm_provider"] == "openai_compatible"
    assert config["backend_url"] == "http://localhost:1234/v1"
    assert config["deep_think_llm"] == "deep"
    assert config["quick_think_llm"] == "quick"
    assert config["output_language"] == "Chinese"
    assert config["max_debate_rounds"] == 2
    assert config["max_risk_discuss_rounds"] == 3


def test_find_tradingagents_python_prefers_env_var(monkeypatch, tmp_path):
    python_path = tmp_path / "bin" / "python"
    python_path.parent.mkdir()
    python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    python_path.chmod(0o755)
    monkeypatch.setenv("TRADINGAGENTS_PYTHON", str(python_path))

    assert find_tradingagents_python(tmp_path) == python_path


def test_run_tradingagents_analysis_uses_subprocess_python(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1\n", encoding="utf-8")
    python_path = tmp_path / "python"
    python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    python_path.chmod(0o755)
    captured = {}

    def fake_find_python(repo_path):
        return python_path

    def fake_run(command, input, text, capture_output, cwd, env, timeout, check):
        captured["command"] = command
        captured["payload"] = json.loads(input)
        payload = {
            "status": "succeeded",
            "symbol": "SZ002241",
            "tradingagents_ticker": "002241.SZ",
            "analysis_date": "2026-07-05",
            "elapsed_seconds": 0.1,
            "reports": {"market_report": "market"},
            "warnings": [],
        }
        return type("Completed", (), {"stdout": "__BACKTEST_TRADINGAGENTS_JSON__" + json.dumps(payload) + "\n"})()

    monkeypatch.setattr(tradingagents_adapter, "find_tradingagents_python", fake_find_python)
    monkeypatch.setattr(tradingagents_adapter.subprocess, "run", fake_run)

    response = run_tradingagents_analysis(
        TradingAgentsAnalysisRequest(symbol="SZ002241", analysis_date="2026-07-05"),
        env_path=env_path,
        repo_path=tmp_path,
    )

    assert captured["command"][0] == str(python_path)
    assert captured["payload"]["symbol"] == "SZ002241"
    assert captured["payload"]["env_path"] == str(env_path)
    assert response.reports.market_report == "market"


def test_run_tradingagents_analysis_sanitizes_subprocess_errors(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1\n"
        "OPENAI_COMPATIBLE_API_KEY=dummy\n",
        encoding="utf-8",
    )
    python_path = tmp_path / "python"
    python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    python_path.chmod(0o755)

    def fake_find_python(repo_path):
        return python_path

    def fake_run(*args, **kwargs):
        raise tradingagents_adapter.subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0],
            stderr="failed with dummy",
        )

    monkeypatch.setattr(tradingagents_adapter, "find_tradingagents_python", fake_find_python)
    monkeypatch.setattr(tradingagents_adapter.subprocess, "run", fake_run)

    with pytest.raises(TradingAgentsAdapterError) as exc_info:
        run_tradingagents_analysis(
            TradingAgentsAnalysisRequest(symbol="SZ002241", analysis_date="2026-07-05"),
            env_path=env_path,
            repo_path=tmp_path,
        )

    assert "dummy" not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)


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
        repo_path=tmp_path,
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
            repo_path=tmp_path,
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
            repo_path=tmp_path,
        )
