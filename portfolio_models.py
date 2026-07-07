from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from optimization_models import AShareTradingConfig
from tradable_universe import TradableUniversePolicy, validate_universe


class UniverseConfig(BaseModel):
    mode: Literal["auto", "manual"] | None = None
    symbols: list[str] = Field(default_factory=list)
    max_symbols: int = 4
    max_scan_symbols: int | None = None
    refresh_universe: bool = False
    blacklist_symbols: list[str] = Field(default_factory=list)
    whitelist_symbols: list[str] = Field(default_factory=list)
    allowed_code_prefixes: tuple[str, ...] = ("60", "00")
    exclude_star: bool = True
    exclude_bj: bool = True
    exclude_funds: bool = True

    @field_validator("max_symbols")
    @classmethod
    def validate_max_symbols(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_symbols must be greater than 0")
        if value > 4:
            raise ValueError("max_symbols must not exceed 4")
        return value

    @field_validator("max_scan_symbols")
    @classmethod
    def validate_max_scan_symbols(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_scan_symbols must be greater than 0")
        return value

    @model_validator(mode="after")
    def validate_symbols(self) -> "UniverseConfig":
        if self.mode is None:
            self.mode = "manual" if self.symbols else "auto"

        self.blacklist_symbols = _normalize_symbol_list(self.blacklist_symbols, self.exclude_funds)
        self.whitelist_symbols = _normalize_symbol_list(self.whitelist_symbols, self.exclude_funds)

        if self.mode == "auto":
            self.symbols = _normalize_symbol_list(self.symbols, self.exclude_funds)
            return self

        if not self.symbols:
            raise ValueError("universe must include at least one symbol")

        policy = TradableUniversePolicy(
            max_symbols=self.max_symbols,
            allowed_code_prefixes=tuple(self.allowed_code_prefixes),
            exclude_funds=self.exclude_funds,
        )
        result = validate_universe(self.symbols, policy=policy)
        blocking_reasons = [
            row.reason or "invalid_symbol"
            for row in result.rejected
            if row.reason != "duplicate_symbol"
        ]
        if blocking_reasons:
            raise ValueError("; ".join(blocking_reasons))

        self.symbols = result.accepted_symbols
        return self


class FactorConfig(BaseModel):
    momentum_lookback: int = 60
    volatility_lookback: int = 20
    liquidity_lookback: int = 20
    momentum_weight: float = 0.45
    volatility_weight: float = -0.25
    liquidity_weight: float = 0.20
    trend_weight: float = 0.10

    @field_validator("momentum_lookback", "volatility_lookback", "liquidity_lookback")
    @classmethod
    def validate_positive_lookback(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("lookback values must be positive")
        return value

    @field_validator("momentum_weight", "volatility_weight", "liquidity_weight", "trend_weight")
    @classmethod
    def validate_finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("factor weights must be finite")
        return value


class SelectionConfig(BaseModel):
    top_n: int = 2
    min_history_bars: int = 120
    min_avg_turnover_value: float | None = None
    min_avg_volume: float | None = None
    min_price: float | None = None
    max_price: float | None = None
    max_data_gap_days: int | None = None
    score_threshold: float | None = None

    @field_validator("top_n", "min_history_bars")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("selection integer values must be positive")
        return value

    @field_validator("top_n")
    @classmethod
    def validate_top_n_cap(cls, value: int) -> int:
        if value > 4:
            raise ValueError("top_n must be fewer than 5")
        return value

    @field_validator("min_avg_turnover_value", "min_avg_volume", "min_price", "max_price")
    @classmethod
    def validate_optional_non_negative_float(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("selection filter values must be non-negative")
        return value

    @field_validator("max_data_gap_days")
    @classmethod
    def validate_optional_non_negative_int(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("max_data_gap_days must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_price_range(self) -> "SelectionConfig":
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("min_price must not exceed max_price")
        return self


class RebalanceConfig(BaseModel):
    frequency: Literal["weekly", "biweekly", "monthly"] = "monthly"
    weekday: int = 0
    monthday: int = 1
    lookahead_safe: bool = True

    @field_validator("weekday")
    @classmethod
    def validate_weekday(cls, value: int) -> int:
        if value < 0 or value > 6:
            raise ValueError("weekday must be between 0 and 6")
        return value

    @field_validator("monthday")
    @classmethod
    def validate_monthday(cls, value: int) -> int:
        if value < 1 or value > 31:
            raise ValueError("monthday must be between 1 and 31")
        return value


class PortfolioRiskConfig(BaseModel):
    max_position_pct: float = 0.50
    target_gross_exposure: float = 0.95
    cash_buffer_pct: float = 0.05
    stop_loss_pct: float | None = None
    max_drawdown_stop_pct: float | None = 30

    @field_validator("max_position_pct", "target_gross_exposure")
    @classmethod
    def validate_pct_within_one(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("percentage values must be within (0, 1]")
        return value

    @field_validator("cash_buffer_pct")
    @classmethod
    def validate_cash_buffer(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("cash_buffer_pct must be within [0, 1]")
        return value


class PortfolioBacktestRequest(BaseModel):
    start_date: str
    end_date: str
    initial_cash: float = 100000
    data_provider: str = "auto"
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    factors: FactorConfig = Field(default_factory=FactorConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    trading: AShareTradingConfig = Field(default_factory=AShareTradingConfig)
    risk: PortfolioRiskConfig = Field(default_factory=PortfolioRiskConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("initial_cash")
    @classmethod
    def validate_initial_cash(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("initial_cash must be greater than 0")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> "PortfolioBacktestRequest":
        start = _parse_date(self.start_date, "start_date")
        end = _parse_date(self.end_date, "end_date")
        if start >= end:
            raise ValueError("start_date must be earlier than end_date")
        if self.universe.mode == "manual" and self.selection.top_n > len(self.universe.symbols):
            raise ValueError("top_n must not exceed symbol count")
        return self


@dataclass
class PortfolioBacktestResult:
    summary: dict[str, Any]
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    rebalance_events: list[dict[str, Any]] = field(default_factory=list)
    candidate_rankings: list[dict[str, Any]] = field(default_factory=list)
    data_warnings: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    scan_diagnostics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_api_response(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "equity_curve": self.equity_curve,
            "positions": self.positions,
            "trades": self.trades,
            "rebalance_events": self.rebalance_events,
            "candidate_rankings": self.candidate_rankings,
            "data_warnings": self.data_warnings,
            "risk_flags": self.risk_flags,
            "scan_diagnostics": self.scan_diagnostics,
            "config": self.config,
        }


def _parse_date(value: str, field_name: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format") from exc


def _normalize_symbol_list(symbols: list[str], exclude_funds: bool) -> list[str]:
    if not symbols:
        return []
    policy = TradableUniversePolicy(max_symbols=max(len(symbols), 1), exclude_funds=exclude_funds)
    result = validate_universe(symbols, policy=policy)
    blocking_reasons = [
        row.reason or "invalid_symbol"
        for row in result.rejected
        if row.reason != "duplicate_symbol"
    ]
    if blocking_reasons:
        raise ValueError("; ".join(blocking_reasons))
    return result.accepted_symbols
