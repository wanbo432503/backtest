#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8005}"
PYTHON_BIN="${PYTHON:-python}"

find_port_pids() {
    lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
}

wait_for_port_free() {
    for _ in $(seq 1 20); do
        if [ -z "$(find_port_pids)" ]; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

stop_existing_server() {
    local pids
    pids="$(find_port_pids)"
    if [ -z "${pids}" ]; then
        echo "No process is listening on port ${PORT}; starting backtest."
        return 0
    fi

    echo "Port ${PORT} is already in use. Restarting these process(es):"
    for pid in ${pids}; do
        ps -p "${pid}" -o pid= -o command= || true
    done

    kill ${pids} 2>/dev/null || true
    if wait_for_port_free; then
        return 0
    fi

    echo "Port ${PORT} did not close after SIGTERM; forcing shutdown."
    pids="$(find_port_pids)"
    if [ -n "${pids}" ]; then
        kill -9 ${pids} 2>/dev/null || true
    fi
    wait_for_port_free
}

if ! command -v lsof >/dev/null 2>&1; then
    echo "lsof is required to detect the current server process." >&2
    exit 1
fi

cd "${ROOT_DIR}"
stop_existing_server

echo "Starting backtest at http://${HOST}:${PORT}"
exec "$PYTHON_BIN" -m uvicorn main:app --host "${HOST}" --port "${PORT}"
