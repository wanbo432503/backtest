from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from market_data import to_yfinance_symbol


InfoLoader = Callable[[str], dict[str, Any]]


class FundamentalsBundle(BaseModel):
    values_by_symbol: dict[str, dict[str, float]] = Field(default_factory=dict)
    requested_symbols: list[str] = Field(default_factory=list)
    loaded_symbols: list[str] = Field(default_factory=list)
    missing_symbols: list[str] = Field(default_factory=list)
    coverage_pct: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    errors_by_symbol: dict[str, str] = Field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "requested_symbols": len(self.requested_symbols),
            "loaded_fundamentals": len(self.loaded_symbols),
            "coverage_pct": self.coverage_pct,
            "missing_symbols": self.missing_symbols,
            "errors_by_symbol": self.errors_by_symbol,
            "warnings": self.warnings,
        }


def load_portfolio_fundamentals(
    symbols: list[str],
    *,
    data_provider: str = "auto",
    info_loader: InfoLoader | None = None,
    min_coverage_pct: float = 50.0,
) -> FundamentalsBundle:
    requested_symbols = list(dict.fromkeys(symbols))
    if not requested_symbols:
        return FundamentalsBundle()

    loader = info_loader or _load_yfinance_info
    values_by_symbol: dict[str, dict[str, float]] = {}
    loaded_symbols: list[str] = []
    missing_symbols: list[str] = []
    errors_by_symbol: dict[str, str] = {}

    for symbol in requested_symbols:
        yfinance_symbol = to_yfinance_symbol(symbol)
        try:
            info = loader(yfinance_symbol)
        except Exception as exc:
            info = {}
            errors_by_symbol[symbol] = str(exc)

        values = _extract_value_quality_factors(info)
        if values:
            values_by_symbol[symbol] = values
            loaded_symbols.append(symbol)
        else:
            missing_symbols.append(symbol)

    coverage_pct = len(loaded_symbols) / len(requested_symbols) * 100
    warnings = []
    if coverage_pct < min_coverage_pct:
        warnings.append(
            f"fundamental coverage low: {coverage_pct:.1f}% "
            f"({len(loaded_symbols)}/{len(requested_symbols)})"
        )
    if data_provider == "mootdx" and loaded_symbols:
        warnings.append("mootdx does not provide fundamentals; yfinance fallback was used")

    return FundamentalsBundle(
        values_by_symbol=values_by_symbol,
        requested_symbols=requested_symbols,
        loaded_symbols=loaded_symbols,
        missing_symbols=missing_symbols,
        coverage_pct=coverage_pct,
        warnings=warnings,
        errors_by_symbol=errors_by_symbol,
    )


def _load_yfinance_info(yfinance_symbol: str) -> dict[str, Any]:
    import yfinance as yf

    return dict(yf.Ticker(yfinance_symbol).info or {})


def _extract_value_quality_factors(info: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    pe = _first_finite(info, "trailingPE", "forwardPE")
    pb = _first_finite(info, "priceToBook")
    roe = _first_finite(info, "returnOnEquity")
    revenue_growth = _first_finite(info, "revenueGrowth")
    profit_growth = _first_finite(info, "earningsGrowth", "netIncomeGrowth")

    if pe is not None and pe > 0:
        values["pe_inverse"] = 1 / pe
    if pb is not None and pb > 0:
        values["pb_inverse"] = 1 / pb
    if roe is not None:
        values["roe"] = roe
    if revenue_growth is not None:
        values["revenue_growth"] = revenue_growth
    if profit_growth is not None:
        values["profit_growth"] = profit_growth
    return values


def _first_finite(info: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            return numeric_value
    return None
