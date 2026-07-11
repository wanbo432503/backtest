import json
import re
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tradingagents_config import (
    TRADINGAGENTS_ENV_PATH,
    TRADINGAGENTS_PROJECT_PATH,
    parse_env_file,
)
from tradingagents_models import (
    ALLOWED_ANALYSTS,
    TradingAgentsAnalysisRequest,
    TradingAgentsAnalysisResponse,
    TradingAgentsPortfolioSummaryRequest,
    TradingAgentsPortfolioSummaryResponse,
    TradingAgentsReports,
)


A_SHARE_PREFIX_PATTERN = re.compile(r"^(SH|SZ|BJ)(\d{6})$")
A_SHARE_SUFFIX_PATTERN = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$")
A_SHARE_NUMERIC_PATTERN = re.compile(r"^\d{6}$")
PORTFOLIO_SUMMARY_WARNING = "AI summary is explanatory only and is not used by backtest metrics."


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


def build_run_config(
    request: TradingAgentsAnalysisRequest,
    env_values: dict[str, str],
    project_path: Path = TRADINGAGENTS_PROJECT_PATH,
    base_config: dict | None = None,
) -> dict:
    config = dict(base_config or {})
    home = Path.home() / ".tradingagents"
    config.setdefault("project_dir", str(project_path))
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
    project_path: Path = TRADINGAGENTS_PROJECT_PATH,
    graph_class=None,
) -> TradingAgentsAnalysisResponse:
    selected_analysts = validate_analysts(request.analysts)
    normalized_symbol = normalize_a_share_symbol(request.symbol)
    ticker = to_tradingagents_ticker(normalized_symbol)

    env_values = load_tradingagents_env(env_path)
    started_at = monotonic()

    try:
        with temporary_environ(env_values):
            base_config = None if graph_class is not None else _load_default_config()
            graph_cls = graph_class or _load_tradingagents_graph_class()
            config = build_run_config(
                request,
                env_values,
                project_path=project_path,
                base_config=base_config,
            )
            graph = graph_cls(selected_analysts, config=config, debug=True)
            final_state = _coerce_final_state(
                graph.propagate(ticker, request.analysis_date, asset_type="stock")
            )
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


def build_portfolio_summary_prompt(request: TradingAgentsPortfolioSummaryRequest) -> str:
    payload = {
        "selected_symbols": request.selected_symbols,
        "summary_metrics": request.summary_metrics,
        "latest_candidate_rankings": request.latest_candidate_rankings[-20:],
        "risk_flags": request.risk_flags[:20],
        "scan_diagnostics": request.scan_diagnostics,
    }
    payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
    return (
        "你是一个面向个人投资者的 A 股量化组合复盘助手。\n"
        "请基于下面这次可复现组合回测输出，生成中文解释。\n"
        "只解释选股、调仓、风险和指标含义；不要改写、重算或补造任何回测指标。\n"
        "不要给出确定性收益承诺，不要把解释当成交易指令。\n"
        "输出建议包含: 1) 组合概览 2) 入选原因 3) 主要风险 4) 下一次人工复核清单。\n\n"
        f"回测输出 JSON:\n{payload_text}"
    )


def run_tradingagents_portfolio_summary(
    request: TradingAgentsPortfolioSummaryRequest,
    env_path: Path = TRADINGAGENTS_ENV_PATH,
    completion_client: Callable[[str, dict[str, str]], str] | None = None,
) -> TradingAgentsPortfolioSummaryResponse:
    env_values = load_tradingagents_env(env_path)
    prompt = build_portfolio_summary_prompt(request)
    started_at = monotonic()
    active_client = completion_client or _call_openai_compatible_chat

    try:
        summary_text = str(active_client(prompt, env_values)).strip()
        if not summary_text:
            raise ValueError("portfolio summary response is empty")
    except Exception as exc:  # noqa: BLE001 - sanitize third-party LLM/backend errors
        raise TradingAgentsAdapterError(sanitize_error_message(str(exc), env_values)) from exc

    return TradingAgentsPortfolioSummaryResponse(
        summary_text=sanitize_error_message(summary_text, env_values),
        elapsed_seconds=round(monotonic() - started_at, 3),
        warnings=[PORTFOLIO_SUMMARY_WARNING],
    )


def _call_openai_compatible_chat(prompt: str, env_values: dict[str, str]) -> str:
    backend_url = env_values.get("TRADINGAGENTS_LLM_BACKEND_URL")
    if not backend_url:
        raise ValueError("TRADINGAGENTS_LLM_BACKEND_URL is required for portfolio summary")

    model = env_values.get("TRADINGAGENTS_DEEP_THINK_LLM") or env_values.get("TRADINGAGENTS_QUICK_THINK_LLM")
    if not model:
        raise ValueError("TRADINGAGENTS_DEEP_THINK_LLM or TRADINGAGENTS_QUICK_THINK_LLM is required")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是谨慎的 A 股量化回测解释助手，只解释输入，不改写回测结果。",
            },
            {"role": "user", "content": prompt},
        ],
    }
    if env_values.get("TRADINGAGENTS_TEMPERATURE"):
        payload["temperature"] = float(env_values["TRADINGAGENTS_TEMPERATURE"])

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = env_values.get("OPENAI_COMPATIBLE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(_chat_completions_url(backend_url), data=body, headers=headers, method="POST")
    timeout = float(env_values.get("TRADINGAGENTS_PORTFOLIO_SUMMARY_TIMEOUT_SECONDS", "120"))
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured local/compatible LLM URL
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM backend returned {exc.code}: {error_body}") from exc

    choices = response_data.get("choices") or []
    if not choices:
        raise ValueError("LLM backend returned no choices")
    message = choices[0].get("message") or {}
    content = str(message.get("content") or "").strip()
    if not content:
        raise ValueError("LLM backend returned empty content")
    return content


def _chat_completions_url(backend_url: str) -> str:
    url = backend_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _coerce_final_state(result) -> dict:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and result and isinstance(result[0], dict):
        return result[0]
    raise TypeError(
        f"TradingAgents returned unsupported final state type: {type(result).__name__}"
    )


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
