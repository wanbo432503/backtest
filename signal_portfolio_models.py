from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from optimization_models import AShareTradingConfig
from portfolio_models import FactorConfig, SelectionConfig
from tradable_universe import TradableUniversePolicy, validate_universe


class SignalUniverseConfig(BaseModel):
    mode: Literal["auto", "manual"] = "manual"
    symbols: list[str] = Field(default_factory=list)
    max_symbols: int = 50
    max_scan_symbols: int | None = 30
    ohlcv_batch_size: int = 20
    ohlcv_batch_delay_seconds: float = 0.0
    ohlcv_request_delay_seconds: float = 0.0
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
        if value < 1 or value > 50:
            raise ValueError("max_symbols must be between 1 and 50")
        return value

    @field_validator("max_scan_symbols")
    @classmethod
    def validate_scan_limit(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_scan_symbols must be greater than 0")
        return value

    @field_validator("ohlcv_batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("ohlcv_batch_size must be greater than 0")
        return value

    @field_validator("ohlcv_batch_delay_seconds", "ohlcv_request_delay_seconds")
    @classmethod
    def validate_delay(cls, value: float) -> float:
        if value < 0:
            raise ValueError("rate limit delays must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_manual_symbols(self) -> "SignalUniverseConfig":
        if self.mode == "auto":
            self.symbols = []
            return self
        if not self.symbols:
            raise ValueError("manual signal universe must include at least one symbol")
        result = validate_universe(
            self.symbols,
            policy=TradableUniversePolicy(
                max_symbols=self.max_symbols,
                allowed_code_prefixes=self.allowed_code_prefixes,
                exclude_funds=self.exclude_funds,
            ),
        )
        blocking = [row.reason for row in result.rejected if row.reason != "duplicate_symbol"]
        if blocking:
            raise ValueError("; ".join(str(reason) for reason in blocking))
        self.symbols = result.accepted_symbols
        return self


class TrendPullbackPinBarSignalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_name: Literal["trend_pullback_pin_bar"] = "trend_pullback_pin_bar"
    short_ma_period: int = 20
    medium_ma_period: int = 60
    long_ma_period: int = 120
    ma_distance_pct: float = 2.0
    support_lookback: int = 20
    support_tolerance_pct: float = 1.0
    lower_shadow_body_ratio: float = 2.5
    max_body_range_pct: float = 30.0
    min_close_location_pct: float = 65.0
    max_upper_shadow_range_pct: float = 20.0
    volume_lookback: int = 20
    volume_multiplier: float = 1.3
    atr_period: int = 14
    min_stop_distance_pct: float = 1.5
    max_stop_distance_pct: float = 6.0
    reward_risk_ratio: float = 2.5
    max_entry_gap_pct: float = 2.0
    risk_per_trade_pct: float = 0.5
    market_breadth_threshold_pct: float = 50.0
    trend_exit_confirmation_days: Literal[2] = 2
    cooldown_days: int = 20
    price_tick: Literal[0.01] = 0.01

    @model_validator(mode="after")
    def validate_parameters(self) -> "TrendPullbackPinBarSignalConfig":
        periods = [
            self.short_ma_period,
            self.medium_ma_period,
            self.long_ma_period,
        ]
        if periods != sorted(periods) or len(set(periods)) != 3:
            raise ValueError("moving-average periods must be strictly increasing")
        if min(periods) < 2 or min(self.support_lookback, self.volume_lookback, self.atr_period) < 2:
            raise ValueError("strategy lookback periods must be at least 2")
        if not 0 < self.ma_distance_pct <= 20 or not 0 < self.support_tolerance_pct <= 20:
            raise ValueError("pullback distance percentages must be within (0, 20]")
        if self.lower_shadow_body_ratio < 1:
            raise ValueError("lower_shadow_body_ratio must be at least 1")
        bounded_percentages = [
            self.max_body_range_pct,
            self.min_close_location_pct,
            self.max_upper_shadow_range_pct,
            self.min_stop_distance_pct,
            self.max_stop_distance_pct,
            self.max_entry_gap_pct,
            self.risk_per_trade_pct,
            self.market_breadth_threshold_pct,
        ]
        if any(value <= 0 or value > 100 for value in bounded_percentages):
            raise ValueError("strategy percentages must be within (0, 100]")
        if self.min_stop_distance_pct >= self.max_stop_distance_pct:
            raise ValueError("min_stop_distance_pct must be below max_stop_distance_pct")
        if not 1 <= self.volume_multiplier <= 10:
            raise ValueError("volume_multiplier must be between 1 and 10")
        if not 2 <= self.reward_risk_ratio <= 3:
            raise ValueError("reward_risk_ratio must be between 2 and 3")
        if self.cooldown_days < 0 or self.cooldown_days > 252:
            raise ValueError("cooldown_days must be between 0 and 252")
        return self


class SignalPortfolioRiskConfig(BaseModel):
    max_positions: int = 10
    max_position_pct: float = 0.10
    target_gross_exposure: float = 0.85
    max_drawdown_stop_pct: float | None = 30.0

    @field_validator("max_positions")
    @classmethod
    def validate_max_positions(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError("max_positions must be between 1 and 20")
        return value

    @field_validator("max_position_pct", "target_gross_exposure")
    @classmethod
    def validate_percentage(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("portfolio percentages must be within (0, 1]")
        return value


class SignalPortfolioBacktestRequest(BaseModel):
    start_date: str
    end_date: str
    initial_cash: float = 100000
    data_provider: str = "auto"
    universe: SignalUniverseConfig
    strategy: TrendPullbackPinBarSignalConfig = Field(
        default_factory=TrendPullbackPinBarSignalConfig
    )
    trading: AShareTradingConfig = Field(default_factory=AShareTradingConfig)
    risk: SignalPortfolioRiskConfig = Field(default_factory=SignalPortfolioRiskConfig)
    selection: SelectionConfig = Field(
        default_factory=lambda: SelectionConfig(top_n=10, min_history_bars=150)
    )
    factors: FactorConfig = Field(default_factory=FactorConfig)
    selection_strategy: None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("initial_cash")
    @classmethod
    def validate_cash(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("initial_cash must be greater than 0")
        return value

    @model_validator(mode="after")
    def sync_internal_selection(self) -> "SignalPortfolioBacktestRequest":
        try:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date)
        except ValueError as exc:
            raise ValueError("dates must use YYYY-MM-DD format") from exc
        if start >= end:
            raise ValueError("start_date must be earlier than end_date")
        self.selection.top_n = self.risk.max_positions
        return self


@dataclass
class SignalPortfolioBacktestResult:
    summary: dict[str, Any]
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    symbol_contributions: list[dict[str, Any]] = field(default_factory=list)
    signal_events: list[dict[str, Any]] = field(default_factory=list)
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
            "symbol_contributions": self.symbol_contributions,
            "signal_events": self.signal_events,
            "data_warnings": self.data_warnings,
            "risk_flags": self.risk_flags,
            "scan_diagnostics": self.scan_diagnostics,
            "config": self.config,
        }
