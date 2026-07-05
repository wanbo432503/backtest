import re
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from tradingagents_config import TRADINGAGENTS_ENV_PATH, TRADINGAGENTS_REPO_PATH, parse_env_file
from tradingagents_models import (
    ALLOWED_ANALYSTS,
    TradingAgentsAnalysisRequest,
    TradingAgentsAnalysisResponse,
    TradingAgentsReports,
)


A_SHARE_PREFIX_PATTERN = re.compile(r"^(SH|SZ|BJ)(\d{6})$")
A_SHARE_SUFFIX_PATTERN = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$")
A_SHARE_NUMERIC_PATTERN = re.compile(r"^\d{6}$")
SUBPROCESS_JSON_PREFIX = "__BACKTEST_TRADINGAGENTS_JSON__"


class TradingAgentsAdapterError(Exception):
    pass


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


def sanitize_error_message(message: str, env_values: dict[str, str]) -> str:
    sanitized = message
    for key, value in env_values.items():
        if not value:
            continue
        if key.endswith("_API_KEY") or "SECRET" in key or "TOKEN" in key:
            sanitized = sanitized.replace(value, "<redacted>")
    return sanitized


@contextmanager
def temporary_sys_path(path: Path):
    original = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = original


@contextmanager
def temporary_environ(overrides: dict[str, str]):
    original = os.environ.copy()
    os.environ.update({key: value for key, value in overrides.items() if value is not None})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def load_tradingagents_env(env_path: Path = TRADINGAGENTS_ENV_PATH) -> dict[str, str]:
    _, values = parse_env_file(env_path)
    return values


def find_tradingagents_python(repo_path: Path = TRADINGAGENTS_REPO_PATH) -> Path | None:
    configured = os.environ.get("TRADINGAGENTS_PYTHON")
    candidates = [Path(configured)] if configured else []
    candidates.append(repo_path / ".venv" / "bin" / "python")
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def build_run_config(
    request: TradingAgentsAnalysisRequest,
    env_values: dict[str, str],
    repo_path: Path = TRADINGAGENTS_REPO_PATH,
    base_config: dict | None = None,
) -> dict:
    config = dict(base_config or {})
    home = Path.home() / ".tradingagents"
    config.setdefault("project_dir", str(repo_path))
    config.setdefault("results_dir", str(home / "logs"))
    config.setdefault("data_cache_dir", str(home / "cache"))
    config.setdefault("memory_log_path", str(home / "memory" / "trading_memory.md"))
    config.setdefault("max_recur_limit", 100)
    config.setdefault("checkpoint_enabled", False)

    config["llm_provider"] = env_values.get("TRADINGAGENTS_LLM_PROVIDER", "openai_compatible")
    config["backend_url"] = env_values.get("TRADINGAGENTS_LLM_BACKEND_URL")
    config["deep_think_llm"] = env_values.get("TRADINGAGENTS_DEEP_THINK_LLM")
    config["quick_think_llm"] = env_values.get("TRADINGAGENTS_QUICK_THINK_LLM")
    config["output_language"] = env_values.get("TRADINGAGENTS_OUTPUT_LANGUAGE", "Chinese")
    config["max_debate_rounds"] = request.max_debate_rounds
    config["max_risk_discuss_rounds"] = request.max_risk_rounds

    if env_values.get("TRADINGAGENTS_CHECKPOINT_ENABLED"):
        config["checkpoint_enabled"] = _bool_env(env_values["TRADINGAGENTS_CHECKPOINT_ENABLED"])
    if env_values.get("TRADINGAGENTS_TEMPERATURE"):
        config["temperature"] = float(env_values["TRADINGAGENTS_TEMPERATURE"])
    if env_values.get("TRADINGAGENTS_OPENAI_REASONING_EFFORT"):
        config["openai_reasoning_effort"] = env_values["TRADINGAGENTS_OPENAI_REASONING_EFFORT"]
    return config


def run_tradingagents_analysis(
    request: TradingAgentsAnalysisRequest,
    env_path: Path = TRADINGAGENTS_ENV_PATH,
    repo_path: Path = TRADINGAGENTS_REPO_PATH,
    graph_class=None,
) -> TradingAgentsAnalysisResponse:
    selected_analysts = validate_analysts(request.analysts)
    normalized_symbol = normalize_a_share_symbol(request.symbol)
    ticker = to_tradingagents_ticker(normalized_symbol)

    if graph_class is None and not os.environ.get("TRADINGAGENTS_DISABLE_SUBPROCESS"):
        python_path = find_tradingagents_python(repo_path)
        if python_path is not None:
            return _run_tradingagents_analysis_subprocess(request, env_path, repo_path, python_path)

    env_values = load_tradingagents_env(env_path)
    started_at = monotonic()

    try:
        with temporary_sys_path(repo_path), temporary_environ(env_values):
            base_config = None if graph_class is not None else _load_default_config()
            graph_cls = graph_class or _load_tradingagents_graph_class()
            config = build_run_config(request, env_values, repo_path=repo_path, base_config=base_config)
            graph = graph_cls(selected_analysts, config=config, debug=True)
            final_state = graph.propagate(ticker, request.analysis_date, asset_type="stock")
    except Exception as exc:  # noqa: BLE001 - surface sanitized adapter errors to API layer
        raise TradingAgentsAdapterError(sanitize_error_message(str(exc), env_values)) from exc

    return TradingAgentsAnalysisResponse(
        status="succeeded",
        symbol=normalized_symbol,
        tradingagents_ticker=ticker,
        analysis_date=request.analysis_date,
        elapsed_seconds=round(monotonic() - started_at, 3),
        reports=extract_reports(final_state or {}),
        warnings=[],
    )


def _run_tradingagents_analysis_subprocess(
    request: TradingAgentsAnalysisRequest,
    env_path: Path,
    repo_path: Path,
    python_path: Path,
) -> TradingAgentsAnalysisResponse:
    env_values = load_tradingagents_env(env_path)
    payload = request.model_dump()
    payload["env_path"] = str(env_path)
    payload["repo_path"] = str(repo_path)
    command = [str(python_path), str(Path(__file__).parent / "scripts" / "run_tradingagents_analysis.py")]
    env = os.environ.copy()
    env["TRADINGAGENTS_DISABLE_SUBPROCESS"] = "1"

    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            cwd=str(Path(__file__).parent),
            env=env,
            timeout=900,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise TradingAgentsAdapterError(sanitize_error_message(message, env_values)) from exc
    except subprocess.TimeoutExpired as exc:
        raise TradingAgentsAdapterError("TradingAgents analysis timed out") from exc

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(SUBPROCESS_JSON_PREFIX):
            return TradingAgentsAnalysisResponse.model_validate_json(
                line.removeprefix(SUBPROCESS_JSON_PREFIX)
            )
    raise TradingAgentsAdapterError("TradingAgents subprocess did not return a JSON response")


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


def _bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_default_config() -> dict:
    from tradingagents.default_config import DEFAULT_CONFIG

    return dict(DEFAULT_CONFIG)


def _load_tradingagents_graph_class():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    return TradingAgentsGraph
