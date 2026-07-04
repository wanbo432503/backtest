import math
from typing import Any

import pandas as pd


def parse_percent(value: str | int | float | None) -> float:
    """Parse backtesting.py percent-like values into plain percentage numbers."""
    if value is None:
        return 0.0
    if isinstance(value, str):
        value = value.strip().replace("%", "").replace(",", "")
        if not value:
            return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def calculate_score(cagr_pct: float, sharpe: float, max_drawdown_pct: float) -> float:
    """Phase 2.0 score: return and Sharpe rewarded, drawdown penalized."""
    safe_sharpe = 0.0 if sharpe is None or math.isnan(float(sharpe)) else float(sharpe)
    return float(cagr_pct) * 0.4 + safe_sharpe * 0.3 - abs(float(max_drawdown_pct)) * 0.3


def extract_core_metrics(
    stats: pd.Series | dict[str, Any],
    min_trades: int = 5,
    max_drawdown_limit_pct: float = 30,
) -> dict[str, Any]:
    values = stats.to_dict() if isinstance(stats, pd.Series) else dict(stats)

    annual_return_pct = _first_percent(
        values,
        "Return (Ann.) [%]",
        "CAGR [%]",
        "年复合增长率",
        "annual_return_pct",
    )
    sharpe = _first_number(values, "Sharpe Ratio", "夏普比率", "sharpe")
    max_drawdown_pct = abs(
        _first_percent(values, "Max. Drawdown [%]", "最大回撤", "max_drawdown_pct")
    )
    trades = int(_first_number(values, "# Trades", "交易次数", "trades"))
    score = calculate_score(annual_return_pct, sharpe, max_drawdown_pct)

    risk_notes = []
    if trades < min_trades:
        risk_notes.append("交易次数不足")
    if max_drawdown_pct > max_drawdown_limit_pct:
        risk_notes.append("最大回撤超过阈值")

    return {
        "annual_return_pct": annual_return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "trades": trades,
        "score": score,
        "is_rankable": trades >= min_trades,
        "is_high_risk": max_drawdown_pct > max_drawdown_limit_pct,
        "risk_notes": risk_notes,
    }


def _first_percent(values: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in values:
            return parse_percent(values[key])
    return 0.0


def _first_number(values: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in values:
            return parse_percent(values[key])
    return 0.0
