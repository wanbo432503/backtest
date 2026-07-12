from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from optimization_models import AShareTradingConfig
from portfolio_models import FactorConfig, SelectionConfig
from tradable_universe import TradableUniversePolicy, validate_universe

if TYPE_CHECKING:
    from strategy_library import StrategyLibrary


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


class SignalPortfolioStrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_name: str = "trend_pullback_pin_bar"
    parameters: dict[str, Any] = Field(default_factory=dict)


class SignalMarketFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    breadth_ma_period: int = Field(default=60, ge=2, le=250)
    market_breadth_min_pct: float = 40.0
    market_breadth_threshold_pct: float = 50.0
    market_breadth_partial_risk_pct: float = 50.0

    @model_validator(mode="after")
    def validate_parameters(self) -> "SignalMarketFilterConfig":
        percentages = [
            self.market_breadth_min_pct,
            self.market_breadth_threshold_pct,
            self.market_breadth_partial_risk_pct,
        ]
        if any(value <= 0 or value > 100 for value in percentages):
            raise ValueError("market breadth percentages must be within (0, 100]")
        if self.market_breadth_min_pct >= self.market_breadth_threshold_pct:
            raise ValueError("market_breadth_min_pct must be below the full-risk threshold")
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
    strategy: SignalPortfolioStrategyConfig = Field(
        default_factory=SignalPortfolioStrategyConfig
    )
    market_filter: SignalMarketFilterConfig = Field(default_factory=SignalMarketFilterConfig)
    trading: AShareTradingConfig = Field(default_factory=AShareTradingConfig)
    risk: SignalPortfolioRiskConfig = Field(default_factory=SignalPortfolioRiskConfig)
    selection: SelectionConfig = Field(
        default_factory=lambda: SelectionConfig(top_n=10, min_history_bars=150)
    )
    factors: FactorConfig = Field(default_factory=FactorConfig)
    selection_strategy: None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_strategy_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        strategy = payload.get("strategy")
        if not isinstance(strategy, dict) or "parameters" in strategy:
            return payload

        strategy = dict(strategy)
        breadth_fields = {
            "market_breadth_min_pct",
            "market_breadth_threshold_pct",
            "market_breadth_partial_risk_pct",
        }
        legacy_filter = {
            key: strategy.pop(key)
            for key in list(strategy)
            if key in breadth_fields
        }
        if legacy_filter and "market_filter" in payload:
            raise ValueError(
                "legacy market breadth fields cannot be combined with market_filter"
            )
        if legacy_filter:
            payload["market_filter"] = legacy_filter
        strategy_name = strategy.pop("strategy_name", "trend_pullback_pin_bar")
        payload["strategy"] = {
            "strategy_name": strategy_name,
            "parameters": strategy,
        }
        return payload

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

    def normalized_for_library(
        self,
        library: "StrategyLibrary",
    ) -> "SignalPortfolioBacktestRequest":
        definition = library.get(self.strategy.strategy_name)
        if "signal_portfolio" not in definition.supported_modes:
            raise ValueError(
                f"strategy '{definition.strategy_id}' does not support signal portfolios"
            )
        config = library.validate_config(
            definition.strategy_id,
            self.strategy.parameters,
        )
        normalized = self.model_copy(deep=True)
        normalized.strategy.parameters = config.model_dump(mode="json")
        normalized.selection.min_history_bars = max(
            normalized.selection.min_history_bars,
            definition.min_history_bars(config),
            definition.portfolio_priority_history_bars,
            normalized.market_filter.breadth_ma_period,
        )
        return normalized


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
