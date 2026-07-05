#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRADINGAGENTS_REPO="${TRADINGAGENTS_REPO:-/Users/wanbo/knowledge/knowledge/repo/TradingAgents}"
PYTHON_BIN="${PYTHON:-python}"

if [ ! -d "${TRADINGAGENTS_REPO}" ]; then
    echo "TradingAgents repo not found: ${TRADINGAGENTS_REPO}" >&2
    exit 1
fi

cd "${ROOT_DIR}"

echo "Using Python:"
"${PYTHON_BIN}" -c 'import sys; print(sys.executable)'

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m pip install -r requirements-tradingagents-openai-compatible.txt
"${PYTHON_BIN}" -m pip install --no-deps -e "${TRADINGAGENTS_REPO}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util

required = [
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
    "tradingagents",
    "typer",
]

missing = []
for module_name in required:
    try:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    except ModuleNotFoundError:
        missing.append(module_name)

if missing:
    raise SystemExit("Missing dependencies after install: " + ", ".join(missing))

print("Dependency check passed.")
PY

echo "All dependencies are installed in the current Python environment."
