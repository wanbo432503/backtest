from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class StrategyFactorSpec(BaseModel):
    key: str
    label: str
    direction: Literal["higher_better", "lower_better"]
    default_weight: float
    default_lookback: int | None = None
    lookback_candidates: list[int] = Field(default_factory=list)
    weight_candidates: list[float] = Field(default_factory=list)
    required: bool = True

    @field_validator("key", "label")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("strategy factor text fields must not be empty")
        return value.strip()

    @field_validator("default_weight")
    @classmethod
    def validate_default_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("strategy factor weights must be finite")
        return value

    @field_validator("default_lookback")
    @classmethod
    def validate_default_lookback(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("default_lookback must be positive")
        return value

    @field_validator("lookback_candidates")
    @classmethod
    def validate_lookback_candidates(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value):
            raise ValueError("lookback candidates must be positive")
        return value

    @field_validator("weight_candidates")
    @classmethod
    def validate_weight_candidates(cls, value: list[float]) -> list[float]:
        if any(not math.isfinite(float(item)) for item in value):
            raise ValueError("weight candidates must be finite")
        return value


class PortfolioSelectionStrategyDefinition(BaseModel):
    strategy_id: str
    name: str
    description: str
    suitable_for: str
    caveats: list[str] = Field(default_factory=list)
    default_rebalance_frequency: Literal["weekly", "biweekly", "monthly"] = "monthly"
    factors: list[StrategyFactorSpec]
    default_top_n: int = 5
    top_n_candidates: list[int] = Field(default_factory=lambda: [3, 5, 10])
    score_threshold_candidates: list[float | None] = Field(default_factory=lambda: [None])

    @field_validator("strategy_id", "name", "description", "suitable_for")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("strategy text fields must not be empty")
        return value.strip()

    @field_validator("factors")
    @classmethod
    def validate_factors(cls, value: list[StrategyFactorSpec]) -> list[StrategyFactorSpec]:
        if not value:
            raise ValueError("strategy definitions must include factors")
        return value

    @field_validator("default_top_n")
    @classmethod
    def validate_default_top_n(cls, value: int) -> int:
        if value <= 0 or value > 20:
            raise ValueError("default_top_n must be between 1 and 20")
        return value

    @field_validator("top_n_candidates")
    @classmethod
    def validate_top_n_candidates(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("top_n_candidates must not be empty")
        if any(item <= 0 or item > 20 for item in value):
            raise ValueError("top_n_candidates must be between 1 and 20")
        return value

    @field_validator("score_threshold_candidates")
    @classmethod
    def validate_score_threshold_candidates(
        cls,
        value: list[float | None],
    ) -> list[float | None]:
        if not value:
            raise ValueError("score_threshold_candidates must not be empty")
        finite_values = [item for item in value if item is not None]
        if any(not math.isfinite(float(item)) for item in finite_values):
            raise ValueError("score_threshold_candidates must be finite")
        return value


class PortfolioSelectionStrategyConfig(BaseModel):
    strategy_id: str = "steady_low_vol_momentum"
    enabled: bool = True
    parameter_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("strategy_id")
    @classmethod
    def validate_strategy_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("strategy_id must not be empty")
        return value.strip()
