from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

import pandas as pd
from pydantic import BaseModel, Field


ParamType = Literal["int", "float", "str", "bool"]
ParamValue = int | float | str | bool
StrategyMode = Literal["single_stock", "signal_portfolio"]


class StrategyParamMeta(BaseModel):
    name: str
    label: str
    type: ParamType
    default: ParamValue
    search_values: list[ParamValue] = Field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    description: str = ""


@dataclass(frozen=True)
class RiskIntent:
    stop_price: float | None = None
    target_price: float | None = None
    risk_per_share: float | None = None
    risk_budget_pct: float | None = None


@dataclass(frozen=True)
class EntryIntent:
    order_type: Literal["next_open", "stop_next_bar"]
    strength: float = 0.0
    trigger_price: float | None = None
    expires_after_bars: int = 1
    suggested_position_pct: float = 1.0
    risk: RiskIntent | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitIntent:
    reason: str
    order_type: Literal["next_open"] = "next_open"


@dataclass(frozen=True)
class SimulationPosition:
    symbol: str
    shares: int
    entry_date: str
    entry_price: float
    entry_cost: float = 0.0
    holding_bars: int = 0
    highest_price: float | None = None
    risk: RiskIntent | None = None


@dataclass(frozen=True)
class StrategyBarContext:
    symbol: str
    frame: pd.DataFrame
    bar_index: int
    config: BaseModel
    position: SimulationPosition | None = None
    state: Mapping[str, Any] = field(default_factory=dict)
    bars_since_exit: int | None = None

    @property
    def history(self) -> pd.DataFrame:
        return self.frame.iloc[: self.bar_index + 1]

    @property
    def current(self) -> pd.Series:
        return self.frame.iloc[self.bar_index]

    @property
    def previous(self) -> pd.Series | None:
        if self.bar_index < 1:
            return None
        return self.frame.iloc[self.bar_index - 1]


@dataclass(frozen=True)
class StrategyDecision:
    entry: EntryIntent | None = None
    exit: ExitIntent | None = None
    next_state: Mapping[str, Any] | None = None


PrepareFrame = Callable[[pd.DataFrame, BaseModel], pd.DataFrame]
EvaluateStrategy = Callable[[StrategyBarContext], StrategyDecision]
MinHistoryBars = Callable[[BaseModel], int]


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    display_name: str
    description: str
    config_model: type[BaseModel]
    parameters: tuple[StrategyParamMeta, ...]
    prepare_frame: PrepareFrame
    evaluate: EvaluateStrategy
    min_history_bars: MinHistoryBars
    supported_modes: tuple[StrategyMode, ...] = (
        "single_stock",
        "signal_portfolio",
    )


@dataclass
class SimulationResult:
    summary: dict[str, Any]
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    signal_events: list[dict[str, Any]] = field(default_factory=list)
    symbol_contributions: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
