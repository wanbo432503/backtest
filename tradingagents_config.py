from pathlib import Path
import importlib.util
import os
import tempfile
from typing import Any

from tradingagents_models import (
    ALLOWED_TRADINGAGENTS_PROVIDER,
    TradingAgentsConfigResponse,
    TradingAgentsConfigTestResponse,
    TradingAgentsConfigUpdate,
    TradingAgentsConfigView,
)


TRADINGAGENTS_REPO_PATH = Path("/Users/wanbo/knowledge/knowledge/repo/TradingAgents")
TRADINGAGENTS_ENV_PATH = TRADINGAGENTS_REPO_PATH / ".env"
TRADINGAGENTS_INSTALL_COMMAND = "./scripts/install_dependencies.sh"
TRADINGAGENTS_REQUIRED_IMPORTS = [
    "backtrader",
    "dotenv",
    "langchain_core",
    "langchain_experimental",
    "langchain_openai",
    "langgraph",
    "langgraph.checkpoint.sqlite",
    "parsel",
    "questionary",
    "redis",
    "rich",
    "stockstats",
    "tqdm",
    "typer",
]

ALLOWED_ENV_KEYS = {
    "TRADINGAGENTS_LLM_PROVIDER",
    "TRADINGAGENTS_LLM_BACKEND_URL",
    "OPENAI_COMPATIBLE_API_KEY",
    "TRADINGAGENTS_DEEP_THINK_LLM",
    "TRADINGAGENTS_QUICK_THINK_LLM",
    "TRADINGAGENTS_OUTPUT_LANGUAGE",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS",
    "TRADINGAGENTS_MAX_RISK_ROUNDS",
    "TRADINGAGENTS_CHECKPOINT_ENABLED",
    "TRADINGAGENTS_TEMPERATURE",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT",
}
SECRET_ENV_KEYS = {"OPENAI_COMPATIBLE_API_KEY"}

FIELD_TO_ENV_KEY = {
    "provider": "TRADINGAGENTS_LLM_PROVIDER",
    "backend_url": "TRADINGAGENTS_LLM_BACKEND_URL",
    "api_key": "OPENAI_COMPATIBLE_API_KEY",
    "deep_model": "TRADINGAGENTS_DEEP_THINK_LLM",
    "quick_model": "TRADINGAGENTS_QUICK_THINK_LLM",
    "output_language": "TRADINGAGENTS_OUTPUT_LANGUAGE",
    "max_debate_rounds": "TRADINGAGENTS_MAX_DEBATE_ROUNDS",
    "max_risk_rounds": "TRADINGAGENTS_MAX_RISK_ROUNDS",
    "checkpoint_enabled": "TRADINGAGENTS_CHECKPOINT_ENABLED",
    "temperature": "TRADINGAGENTS_TEMPERATURE",
    "openai_reasoning_effort": "TRADINGAGENTS_OPENAI_REASONING_EFFORT",
}


def parse_env_file(path: Path) -> tuple[list[str], dict[str, str]]:
    if not path.exists():
        return [], {}

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return lines, values


def get_config_view(env_path: Path = TRADINGAGENTS_ENV_PATH) -> TradingAgentsConfigResponse:
    _, values = parse_env_file(env_path)
    config = TradingAgentsConfigView(
        provider=values.get("TRADINGAGENTS_LLM_PROVIDER", ALLOWED_TRADINGAGENTS_PROVIDER),
        backend_url=_value_or_none(values.get("TRADINGAGENTS_LLM_BACKEND_URL")),
        deep_model=_value_or_none(values.get("TRADINGAGENTS_DEEP_THINK_LLM")),
        quick_model=_value_or_none(values.get("TRADINGAGENTS_QUICK_THINK_LLM")),
        output_language=values.get("TRADINGAGENTS_OUTPUT_LANGUAGE", "Chinese") or "Chinese",
        max_debate_rounds=_int_value(values.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS"), 1),
        max_risk_rounds=_int_value(values.get("TRADINGAGENTS_MAX_RISK_ROUNDS"), 1),
        checkpoint_enabled=_bool_value(values.get("TRADINGAGENTS_CHECKPOINT_ENABLED"), False),
        temperature=_float_value(values.get("TRADINGAGENTS_TEMPERATURE")),
        openai_reasoning_effort=_value_or_none(values.get("TRADINGAGENTS_OPENAI_REASONING_EFFORT")),
        api_key_set=bool(_value_or_none(values.get("OPENAI_COMPATIBLE_API_KEY"))),
    )
    return TradingAgentsConfigResponse(
        repo_path=str(env_path.parent),
        env_path=str(env_path),
        config=config,
    )


def update_config(
    update: TradingAgentsConfigUpdate,
    env_path: Path = TRADINGAGENTS_ENV_PATH,
) -> TradingAgentsConfigResponse:
    lines, _ = parse_env_file(env_path)
    updates = _build_env_updates(update)

    seen: set[str] = set()
    next_lines: list[str] = []
    for line in lines:
        key = _line_key(line)
        if key in updates:
            next_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            next_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            next_lines.append(f"{key}={value}")

    _atomic_write(env_path, "\n".join(next_lines).rstrip() + "\n")
    return get_config_view(env_path=env_path)


def test_config(env_path: Path = TRADINGAGENTS_ENV_PATH) -> TradingAgentsConfigTestResponse:
    _, values = parse_env_file(env_path)
    missing_dependencies = _missing_tradingagents_dependencies()
    checks: list[dict[str, Any]] = [
        {"name": "repo_path", "ok": env_path.parent.exists(), "path": str(env_path.parent)},
        {"name": "env_file", "ok": env_path.exists(), "path": str(env_path)},
        {
            "name": "env_writable",
            "ok": env_path.exists() and os.access(env_path, os.R_OK | os.W_OK),
            "path": str(env_path),
        },
        {
            "name": "provider",
            "ok": values.get("TRADINGAGENTS_LLM_PROVIDER", ALLOWED_TRADINGAGENTS_PROVIDER)
            == ALLOWED_TRADINGAGENTS_PROVIDER,
        },
        {"name": "backend_url", "ok": bool(_value_or_none(values.get("TRADINGAGENTS_LLM_BACKEND_URL")))},
        {"name": "deep_model", "ok": bool(_value_or_none(values.get("TRADINGAGENTS_DEEP_THINK_LLM")))},
        {"name": "quick_model", "ok": bool(_value_or_none(values.get("TRADINGAGENTS_QUICK_THINK_LLM")))},
        {
            "name": "python_dependencies",
            "ok": not missing_dependencies,
            "missing": missing_dependencies,
            "python": "current",
            "install_command": TRADINGAGENTS_INSTALL_COMMAND,
        },
    ]
    warnings = []
    if not _value_or_none(values.get("OPENAI_COMPATIBLE_API_KEY")):
        warnings.append("OPENAI_COMPATIBLE_API_KEY is not set; keyless local endpoints may still work.")
    if missing_dependencies:
        warnings.append(
            "Missing TradingAgents Python dependencies: "
            + ", ".join(missing_dependencies)
            + f". Run `{TRADINGAGENTS_INSTALL_COMMAND}`."
        )

    return TradingAgentsConfigTestResponse(
        ok=all(bool(check["ok"]) for check in checks),
        checks=checks,
        warnings=warnings,
    )


def _build_env_updates(update: TradingAgentsConfigUpdate) -> dict[str, str]:
    updates: dict[str, str] = {}
    payload = update.model_dump()
    for field, env_key in FIELD_TO_ENV_KEY.items():
        value = payload.get(field)
        if field == "api_key":
            if update.clear_api_key:
                updates[env_key] = ""
            elif value:
                updates[env_key] = str(value)
            continue
        if value is None:
            continue
        updates[env_key] = _format_env_value(value)
    return updates


def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _line_key(line: str) -> str | None:
    if not line or line.startswith("#") or "=" not in line:
        return None
    return line.split("=", 1)[0].strip()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp_file:
        tmp_file.write(content)
        tmp_name = tmp_file.name
    Path(tmp_name).replace(path)


def _missing_tradingagents_dependencies() -> list[str]:
    return [
        module_name
        for module_name in TRADINGAGENTS_REQUIRED_IMPORTS
        if _is_missing_module(module_name)
    ]


def _is_missing_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is None
    except ModuleNotFoundError:
        return True


def _value_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int_value(value: str | None, default: int) -> int:
    if not _value_or_none(value):
        return default
    return int(value)


def _float_value(value: str | None) -> float | None:
    if not _value_or_none(value):
        return None
    return float(value)


def _bool_value(value: str | None, default: bool) -> bool:
    if not _value_or_none(value):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
