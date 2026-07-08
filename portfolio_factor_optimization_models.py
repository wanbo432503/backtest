from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from portfolio_models import FactorConfig, PortfolioBacktestRequest, SelectionConfig


class OptimizationSplitConfig(BaseModel):
    method: Literal["ratio", "date"] = "ratio"
    train_ratio: float = 0.7
    validation_start: str | None = None

    @field_validator("train_ratio")
    @classmethod
    def validate_train_ratio(cls, value: float) -> float:
        if value < 0.5 or value > 0.9:
            raise ValueError("train_ratio must be between 0.5 and 0.9")
        return value

    @model_validator(mode="after")
    def validate_validation_start(self) -> "OptimizationSplitConfig":
        if self.method == "date" and not self.validation_start:
            raise ValueError("validation_start is required when split method is date")
        if self.validation_start is not None:
            _parse_date(self.validation_start, "validation_start")
        return self


class FactorSearchSpace(BaseModel):
    momentum_lookback: list[int] = Field(default_factory=lambda: [20, 40, 60, 90, 120])
    volatility_lookback: list[int] = Field(default_factory=lambda: [10, 20, 40])
    liquidity_lookback: list[int] = Field(default_factory=lambda: [10, 20, 40])
    momentum_weight: list[float] = Field(default_factory=lambda: [0.2, 0.35, 0.5, 0.65])
    volatility_weight: list[float] = Field(default_factory=lambda: [-0.5, -0.25, 0])
    liquidity_weight: list[float] = Field(default_factory=lambda: [0, 0.15, 0.3])
    trend_weight: list[float] = Field(default_factory=lambda: [0, 0.1, 0.2])
    top_n: list[int] = Field(default_factory=lambda: [2, 3, 5, 10, 20])
    score_threshold: list[float | None] = Field(default_factory=lambda: [None])

    @field_validator(
        "momentum_lookback",
        "volatility_lookback",
        "liquidity_lookback",
        "momentum_weight",
        "volatility_weight",
        "liquidity_weight",
        "trend_weight",
        "top_n",
        "score_threshold",
    )
    @classmethod
    def validate_non_empty_list(cls, value: list[Any]) -> list[Any]:
        if not value:
            raise ValueError("search-space lists must not be empty")
        return value

    @field_validator("momentum_lookback", "volatility_lookback", "liquidity_lookback")
    @classmethod
    def validate_positive_lookbacks(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value):
            raise ValueError("lookback values must be positive")
        return value

    @field_validator("top_n")
    @classmethod
    def validate_top_n(cls, value: list[int]) -> list[int]:
        if any(item <= 0 or item > 20 for item in value):
            raise ValueError("top_n candidates must be between 1 and 20")
        return value

    @field_validator("momentum_weight", "volatility_weight", "liquidity_weight", "trend_weight")
    @classmethod
    def validate_finite_weights(cls, value: list[float]) -> list[float]:
        if any(not math.isfinite(float(item)) for item in value):
            raise ValueError("weight candidates must be finite")
        return value

    @field_validator("score_threshold")
    @classmethod
    def validate_score_thresholds(cls, value: list[float | None]) -> list[float | None]:
        finite_values = [item for item in value if item is not None]
        if any(not math.isfinite(float(item)) for item in finite_values):
            raise ValueError("score_threshold candidates must be finite")
        return value


class SelectionStrategySearchSpace(BaseModel):
    strategy_id: str
    factor_lookbacks: dict[str, list[int]] = Field(default_factory=dict)
    factor_weights: dict[str, list[float]]
    top_n: list[int] = Field(default_factory=lambda: [3, 5, 10])
    score_threshold: list[float | None] = Field(default_factory=lambda: [None])
    legacy_factor_search_space: FactorSearchSpace | None = None

    @field_validator("strategy_id")
    @classmethod
    def validate_strategy_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("strategy_id must not be empty")
        return value.strip()

    @field_validator("factor_lookbacks")
    @classmethod
    def validate_factor_lookbacks(cls, value: dict[str, list[int]]) -> dict[str, list[int]]:
        for factor_key, candidates in value.items():
            if not factor_key.strip():
                raise ValueError("factor lookback keys must not be empty")
            if not candidates:
                raise ValueError("lookback candidate lists must not be empty")
            if any(item <= 0 for item in candidates):
                raise ValueError("lookback candidates must be positive")
        return value

    @field_validator("factor_weights")
    @classmethod
    def validate_factor_weights(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        if not value:
            raise ValueError("factor_weights must not be empty")
        for factor_key, candidates in value.items():
            if not factor_key.strip():
                raise ValueError("factor weight keys must not be empty")
            if not candidates:
                raise ValueError("factor weight candidate lists must not be empty")
            if any(not math.isfinite(float(item)) for item in candidates):
                raise ValueError("factor weight candidates must be finite")
        return value

    @field_validator("top_n")
    @classmethod
    def validate_strategy_top_n(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("top_n candidates must not be empty")
        if any(item <= 0 or item > 20 for item in value):
            raise ValueError("top_n candidates must be between 1 and 20")
        return value

    @field_validator("score_threshold")
    @classmethod
    def validate_strategy_score_threshold(
        cls,
        value: list[float | None],
    ) -> list[float | None]:
        if not value:
            raise ValueError("score_threshold candidates must not be empty")
        finite_values = [item for item in value if item is not None]
        if any(not math.isfinite(float(item)) for item in finite_values):
            raise ValueError("score_threshold candidates must be finite")
        return value


class PortfolioFactorCandidate(BaseModel):
    candidate_id: str
    factor_config: FactorConfig
    selection_config: SelectionConfig
    selection_strategy_id: str | None = None
    selection_strategy_name: str | None = None
    selection_strategy_overrides: dict[str, Any] = Field(default_factory=dict)


class PortfolioFactorMetrics(BaseModel):
    final_equity: float
    total_return_pct: float
    annual_return_pct: float
    max_drawdown_pct: float
    turnover: float
    rebalances: int
    trades: int
    return_volatility_pct: float = 0.0
    downside_volatility_pct: float = 0.0
    log_equity_trend_r2: float = 0.0
    equity_trend_score: float = 0.0
    positive_return_day_ratio: float = 0.0


class PortfolioFactorOptimizationTrialResult(BaseModel):
    candidate: PortfolioFactorCandidate
    objective_score: float
    train_metrics: PortfolioFactorMetrics
    validation_metrics: PortfolioFactorMetrics
    risk_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PortfolioFactorOptimizationResult(BaseModel):
    best_result: PortfolioFactorOptimizationTrialResult | None = None
    top_results: list[PortfolioFactorOptimizationTrialResult] = Field(default_factory=list)
    split: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PortfolioFactorOptimizationRequest(BaseModel):
    base_request: PortfolioBacktestRequest
    split: OptimizationSplitConfig = Field(default_factory=OptimizationSplitConfig)
    search_space: FactorSearchSpace | None = Field(default_factory=FactorSearchSpace)
    max_trials: int = 200
    max_workers: int = 8
    executor_backend: Literal["process", "thread"] = "process"
    objective: Literal["validation_smooth_uptrend"] = "validation_smooth_uptrend"

    @field_validator("max_trials")
    @classmethod
    def validate_max_trials(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_trials must be greater than 0")
        return value

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, value: int) -> int:
        if value < 1 or value > 8:
            raise ValueError("max_workers must be between 1 and 8")
        return value

    @model_validator(mode="after")
    def validate_search_space_or_strategy(self) -> "PortfolioFactorOptimizationRequest":
        strategy_config = self.base_request.selection_strategy
        has_named_strategy = (
            strategy_config is not None
            and strategy_config.enabled
            and strategy_config.strategy_id != "custom_factor_blend"
        )
        if self.search_space is None and not has_named_strategy:
            raise ValueError("search_space can be null only when a named selection strategy is enabled")
        return self


def _parse_date(value: str, field_name: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format") from exc
