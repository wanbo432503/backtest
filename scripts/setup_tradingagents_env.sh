#!/usr/bin/env bash
set -euo pipefail

TRADINGAGENTS_REPO="${TRADINGAGENTS_REPO:-/Users/wanbo/knowledge/knowledge/repo/TradingAgents}"
TRADINGAGENTS_VENV="${TRADINGAGENTS_VENV:-${TRADINGAGENTS_REPO}/.venv}"

if [ ! -d "${TRADINGAGENTS_REPO}" ]; then
    echo "TradingAgents repo not found: ${TRADINGAGENTS_REPO}" >&2
    exit 1
fi

python -m venv "${TRADINGAGENTS_VENV}"
"${TRADINGAGENTS_VENV}/bin/python" -m pip install --upgrade pip
"${TRADINGAGENTS_VENV}/bin/python" -m pip install -e "${TRADINGAGENTS_REPO}"

echo "TradingAgents environment is ready:"
echo "${TRADINGAGENTS_VENV}/bin/python"
