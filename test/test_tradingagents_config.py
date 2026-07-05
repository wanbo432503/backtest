from pathlib import Path
import importlib.util

import tradingagents_config
from tradingagents_config import get_config_view, parse_env_file, test_config as run_config_check, update_config
from tradingagents_models import TradingAgentsConfigUpdate


def write_env(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_parse_env_file_reads_values_and_keeps_lines(tmp_path):
    env_path = write_env(
        tmp_path / ".env",
        """
        # comment
        TRADINGAGENTS_LLM_PROVIDER=openai_compatible
        TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1
        UNKNOWN_KEY=keep-me
        """,
    )

    lines, values = parse_env_file(env_path)

    assert lines[0] == "# comment"
    assert values["TRADINGAGENTS_LLM_PROVIDER"] == "openai_compatible"
    assert values["UNKNOWN_KEY"] == "keep-me"


def test_get_config_view_masks_api_key(tmp_path):
    env_path = write_env(
        tmp_path / ".env",
        """
        TRADINGAGENTS_LLM_PROVIDER=openai_compatible
        TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1
        TRADINGAGENTS_DEEP_THINK_LLM=deep-model
        TRADINGAGENTS_QUICK_THINK_LLM=quick-model
        OPENAI_COMPATIBLE_API_KEY=secret-value
        """,
    )

    response = get_config_view(env_path=env_path)
    dumped = response.model_dump_json()

    assert response.config.api_key_set is True
    assert response.config.backend_url == "http://localhost:1234/v1"
    assert response.config.deep_model == "deep-model"
    assert "secret-value" not in dumped


def test_update_config_preserves_comments_unknown_keys_and_order(tmp_path):
    env_path = write_env(
        tmp_path / ".env",
        """
        # keep this comment
        UNKNOWN_KEY=keep-me
        TRADINGAGENTS_LLM_BACKEND_URL=http://old.example/v1
        OPENAI_COMPATIBLE_API_KEY=dummy
        """,
    )

    update_config(
        TradingAgentsConfigUpdate(
            backend_url="https://new.example/v1",
            deep_model="deep-model",
            quick_model="quick-model",
            output_language="Chinese",
        ),
        env_path=env_path,
    )

    content = env_path.read_text(encoding="utf-8")
    assert "# keep this comment" in content
    assert "UNKNOWN_KEY=keep-me" in content
    assert "TRADINGAGENTS_LLM_BACKEND_URL=https://new.example/v1" in content
    assert "TRADINGAGENTS_DEEP_THINK_LLM=deep-model" in content
    assert "TRADINGAGENTS_QUICK_THINK_LLM=quick-model" in content
    assert "OPENAI_COMPATIBLE_API_KEY=dummy" in content


def test_update_config_empty_api_key_does_not_overwrite_existing_value(tmp_path):
    env_path = write_env(
        tmp_path / ".env",
        """
        TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1
        OPENAI_COMPATIBLE_API_KEY=dummy
        """,
    )

    update_config(TradingAgentsConfigUpdate(api_key=""), env_path=env_path)

    assert "OPENAI_COMPATIBLE_API_KEY=dummy" in env_path.read_text(encoding="utf-8")


def test_update_config_can_clear_api_key_explicitly(tmp_path):
    env_path = write_env(
        tmp_path / ".env",
        """
        TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1
        OPENAI_COMPATIBLE_API_KEY=dummy
        """,
    )

    response = update_config(TradingAgentsConfigUpdate(clear_api_key=True), env_path=env_path)

    assert response.config.api_key_set is False
    assert "OPENAI_COMPATIBLE_API_KEY=" in env_path.read_text(encoding="utf-8")
    assert "dummy" not in env_path.read_text(encoding="utf-8")


def test_test_config_reports_missing_backend_url(tmp_path):
    env_path = write_env(tmp_path / ".env", "TRADINGAGENTS_LLM_PROVIDER=openai_compatible")

    response = run_config_check(env_path=env_path)

    assert response.ok is False
    assert any(check["name"] == "backend_url" and check["ok"] is False for check in response.checks)


def test_test_config_reports_missing_tradingagents_runtime_dependencies(tmp_path, monkeypatch):
    env_path = write_env(
        tmp_path / ".env",
        """
        TRADINGAGENTS_LLM_PROVIDER=openai_compatible
        TRADINGAGENTS_LLM_BACKEND_URL=http://localhost:1234/v1
        TRADINGAGENTS_DEEP_THINK_LLM=deep-model
        TRADINGAGENTS_QUICK_THINK_LLM=quick-model
        """,
    )

    def fake_find_spec(module_name):
        if module_name == "langchain_core":
            return None
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    response = run_config_check(env_path=env_path)
    dependency_check = next(check for check in response.checks if check["name"] == "python_dependencies")

    assert response.ok is False
    assert dependency_check["ok"] is False
    assert dependency_check["missing"] == ["langchain_core"]
    assert str(tradingagents_config.TRADINGAGENTS_REPO_PATH) in dependency_check["install_command"]


def test_backtest_requirements_do_not_install_tradingagents_into_main_environment():
    content = Path("requirements.txt").read_text(encoding="utf-8")

    assert "-e /Users/wanbo/knowledge/knowledge/repo/TradingAgents" not in content


def test_setup_script_installs_tradingagents_into_dedicated_venv():
    script = Path("scripts/setup_tradingagents_env.sh")

    assert script.exists()
    content = script.read_text(encoding="utf-8")
    assert "python -m venv" in content
    assert "pip install -e" in content
    assert str(tradingagents_config.TRADINGAGENTS_REPO_PATH) in content
