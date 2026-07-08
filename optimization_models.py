from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ParamValue = int | float | str | bool


class RiskConfig(BaseModel):
    enabled: bool = True
    position_pct: float = 0.95
    stop_loss_pct: float | None = 5
    take_profit_pct: float | None = 12
    max_holding_bars: int | None = 120
    cooldown_bars: int = 3
    max_account_drawdown_pct: float | None = 30
    atr_stop_enabled: bool = False
    atr_period: int = 14
    atr_multiplier: float = 2.0

    @field_validator("position_pct")
    @classmethod
    def validate_position_pct(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("position_pct must be within (0, 1]")
        return value


class AShareTradingConfig(BaseModel):
    long_only: bool = True
    t_plus_one: bool = True
    lot_size: int = 100
    limit_up_down_filter: bool = True
    volume_filter: bool = True
    min_volume: float = 1
    slippage_pct: float = 0.05
    buy_commission_pct: float = 0.03
    sell_commission_pct: float = 0.03
    stamp_tax_pct: float = 0.05
    min_commission: float = 5

    @field_validator("lot_size")
    @classmethod
    def validate_lot_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("lot_size must be greater than 0")
        return value


class StrategyParamConfig(BaseModel):
    strategy_name: str
    fixed_params: dict[str, ParamValue] = Field(default_factory=dict)
    search_space: dict[str, list[ParamValue]] = Field(default_factory=dict)


class OptimizationConfig(BaseModel):
    enabled: bool = False
    symbols: list[str] = Field(default_factory=lambda: ["SH603019"])
    strategies: list[StrategyParamConfig] = Field(default_factory=list)
    objective: Literal["score"] = "score"
    top_n: int = 10
    max_combinations: int = 300
    max_workers: int = 8
    min_trades: int = 5
    train_start_date: str | None = None
    train_end_date: str | None = None
    validate_start_date: str | None = None
    validate_end_date: str | None = None
    interval: str = "1d"
    data_provider: str = "auto"

    @field_validator("max_combinations")
    @classmethod
    def validate_max_combinations(cls, value: int) -> int:
        if value > 1000:
            raise ValueError("max_combinations must not exceed 1000")
        if value <= 0:
            raise ValueError("max_combinations must be greater than 0")
        return value

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, value: int) -> int:
        if value < 1 or value > 8:
            raise ValueError("max_workers must be between 1 and 8")
        return value

    @field_validator("symbols")
    @classmethod
    def validate_single_symbol(cls, value: list[str]) -> list[str]:
        if len(value) != 1:
            raise ValueError("symbols must contain exactly one A-share symbol")
        return value


class OptimizationRequest(BaseModel):
    start_date: str
    end_date: str
    strategies: list[StrategyParamConfig] = Field(default_factory=list)
    risk_config: RiskConfig = Field(default_factory=RiskConfig)
    a_share_config: AShareTradingConfig = Field(default_factory=AShareTradingConfig)
    optimization_config: OptimizationConfig = Field(default_factory=OptimizationConfig)
    interval: str = "1d"
    data_provider: str = "auto"
    initial_cash: float = 10000
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_top_level_strategies(self) -> "OptimizationRequest":
        if self.strategies and not self.optimization_config.strategies:
            self.optimization_config.strategies = self.strategies
        if self.interval and self.optimization_config.interval == "1d":
            self.optimization_config.interval = self.interval
        if self.data_provider and self.optimization_config.data_provider == "auto":
            self.optimization_config.data_provider = self.data_provider
        return self
