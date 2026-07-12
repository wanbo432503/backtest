import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.test import SMA
from pydantic import BaseModel, ConfigDict, Field, model_validator

from strategy_engine import (
    EntryIntent,
    ExitIntent,
    StrategyBarContext,
    StrategyDecision,
    StrategyDefinition,
    StrategyParamMeta,
)


class MATrendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fast_ma: int = Field(default=10, ge=2, le=80)
    slow_ma: int = Field(default=30, ge=5, le=180)
    trend_ma: int = Field(default=60, ge=20, le=250)
    momentum_lookback: int = Field(default=5, ge=1, le=60)
    stop_loss_pct: float = Field(default=5, ge=0, le=30)
    take_profit_pct: float = Field(default=12, ge=0, le=80)
    max_holding_bars: int = Field(default=80, ge=1, le=500)
    position_pct: float = Field(default=0.95, ge=0.05, le=1)

    @model_validator(mode="after")
    def validate_periods(self) -> "MATrendConfig":
        if self.fast_ma >= self.slow_ma:
            raise ValueError("fast_ma must be below slow_ma")
        if self.trend_ma < self.slow_ma:
            raise ValueError("trend_ma must be at least slow_ma")
        return self


def crossed_above(previous_fast, previous_slow, current_fast, current_slow) -> bool:
    return previous_fast <= previous_slow and current_fast > current_slow


def crossed_below(previous_fast, previous_slow, current_fast, current_slow) -> bool:
    return previous_fast >= previous_slow and current_fast < current_slow


def should_enter_ma_trend(
    previous_fast_ma,
    previous_slow_ma,
    current_fast_ma,
    current_slow_ma,
    close,
    trend_ma_value,
    momentum_return,
) -> bool:
    values = [
        previous_fast_ma,
        previous_slow_ma,
        current_fast_ma,
        current_slow_ma,
        trend_ma_value,
        momentum_return,
    ]
    if any(np.isnan(float(value)) for value in values):
        return False
    if close <= trend_ma_value:
        return False
    if momentum_return <= 0:
        return False
    return crossed_above(previous_fast_ma, previous_slow_ma, current_fast_ma, current_slow_ma)


def get_ma_exit_reason(
    previous_fast_ma,
    previous_slow_ma,
    current_fast_ma,
    current_slow_ma,
    current_price,
    entry_price,
    holding_bars,
    stop_loss_pct,
    take_profit_pct,
    max_holding_bars,
):
    if crossed_below(previous_fast_ma, previous_slow_ma, current_fast_ma, current_slow_ma):
        return "dead_cross"
    if entry_price is None:
        return None
    if stop_loss_pct and current_price <= entry_price * (1 - stop_loss_pct / 100):
        return "stop_loss"
    if take_profit_pct and current_price >= entry_price * (1 + take_profit_pct / 100):
        return "take_profit"
    if max_holding_bars and holding_bars >= max_holding_bars:
        return "max_holding_bars"
    return None


def prepare_ma_trend_frame(data: pd.DataFrame, config: BaseModel) -> pd.DataFrame:
    resolved = MATrendConfig.model_validate(config.model_dump())
    frame = data.copy()
    close = frame["Close"].astype(float)
    frame["fast_ma_value"] = close.rolling(resolved.fast_ma).mean()
    frame["slow_ma_value"] = close.rolling(resolved.slow_ma).mean()
    frame["trend_ma_value"] = close.rolling(resolved.trend_ma).mean()
    frame["momentum_return"] = close / close.shift(max(1, resolved.momentum_lookback - 1)) - 1
    return frame


def evaluate_ma_trend(context: StrategyBarContext) -> StrategyDecision:
    config = MATrendConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < ma_trend_min_history_bars(config):
        return StrategyDecision()
    previous = context.previous
    if previous is None:
        return StrategyDecision()
    current = context.current
    values = {
        "previous_fast_ma": float(previous["fast_ma_value"]),
        "previous_slow_ma": float(previous["slow_ma_value"]),
        "current_fast_ma": float(current["fast_ma_value"]),
        "current_slow_ma": float(current["slow_ma_value"]),
    }
    current_price = float(current["Close"])
    if context.position is not None:
        reason = get_ma_exit_reason(
            **values,
            current_price=current_price,
            entry_price=context.position.entry_price,
            holding_bars=context.position.holding_bars,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            max_holding_bars=config.max_holding_bars,
        )
        return StrategyDecision(exit=ExitIntent(reason) if reason else None)
    if should_enter_ma_trend(
        **values,
        close=current_price,
        trend_ma_value=float(current["trend_ma_value"]),
        momentum_return=float(current["momentum_return"]),
    ):
        return StrategyDecision(
            entry=EntryIntent(
                order_type="next_open",
                suggested_position_pct=config.position_pct,
                metadata={
                    "stop_loss_pct": config.stop_loss_pct,
                    "take_profit_pct": config.take_profit_pct,
                },
            )
        )
    return StrategyDecision()


def ma_trend_min_history_bars(config: BaseModel) -> int:
    resolved = MATrendConfig.model_validate(config.model_dump())
    return max(resolved.slow_ma, resolved.trend_ma, resolved.momentum_lookback) + 2


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="ma_trend_risk_control",
    display_name="均线趋势风控策略",
    description="短均线上穿长均线且价格位于趋势均线上方时买入，死叉或触发风控时卖出。",
    config_model=MATrendConfig,
    parameters=(
        StrategyParamMeta(name="fast_ma", label="短期均线", type="int", default=10, search_values=[5, 10, 20], min_value=2, max_value=80, step=1),
        StrategyParamMeta(name="slow_ma", label="长期均线", type="int", default=30, search_values=[20, 30, 60], min_value=5, max_value=180, step=1),
        StrategyParamMeta(name="trend_ma", label="趋势均线", type="int", default=60, search_values=[60, 120, 200], min_value=20, max_value=250, step=1),
        StrategyParamMeta(name="momentum_lookback", label="动量窗口", type="int", default=5, search_values=[3, 5, 10], min_value=1, max_value=60, step=1),
        StrategyParamMeta(name="stop_loss_pct", label="止损比例", type="float", default=5, search_values=[3, 5, 8], min_value=0, max_value=30, step=0.5),
        StrategyParamMeta(name="take_profit_pct", label="止盈比例", type="float", default=12, search_values=[8, 12, 20], min_value=0, max_value=80, step=0.5),
        StrategyParamMeta(name="max_holding_bars", label="最大持仓周期", type="int", default=80, search_values=[40, 80, 120], min_value=1, max_value=500, step=1),
        StrategyParamMeta(name="position_pct", label="仓位比例", type="float", default=0.95, search_values=[0.5, 0.8, 0.95], min_value=0.05, max_value=1, step=0.05),
    ),
    prepare_frame=prepare_ma_trend_frame,
    evaluate=evaluate_ma_trend,
    min_history_bars=ma_trend_min_history_bars,
)


class MATrendRiskControlStrategy(Strategy):
    """MA crossover strategy with trend, momentum, and fixed risk controls."""

    strategy_name = "ma_trend_risk_control"
    display_name = "均线趋势风控策略"
    description = "短均线上穿长均线且价格位于趋势均线上方时买入，死叉或触发风控时卖出。"

    fast_ma = 10
    slow_ma = 30
    trend_ma = 60
    momentum_lookback = 5
    stop_loss_pct = 5
    take_profit_pct = 12
    max_holding_bars = 80
    position_pct = 0.95

    def init(self):
        close = self.data.Close
        self.fast = self.I(SMA, close, self.fast_ma)
        self.slow = self.I(SMA, close, self.slow_ma)
        self.trend = self.I(SMA, close, self.trend_ma)
        self.entry_price = None
        self.entry_bar = None

    def next(self):
        min_bars = max(self.slow_ma, self.trend_ma, self.momentum_lookback) + 2
        if len(self.data.Close) < min_bars:
            return

        current_price = float(self.data.Close[-1])
        previous_fast = float(self.fast[-2])
        previous_slow = float(self.slow[-2])
        current_fast = float(self.fast[-1])
        current_slow = float(self.slow[-1])
        trend_value = float(self.trend[-1])
        lookback_price = float(self.data.Close[-self.momentum_lookback])
        momentum_return = current_price / lookback_price - 1

        if self.position:
            holding_bars = len(self.data.Close) - (self.entry_bar or len(self.data.Close))
            reason = get_ma_exit_reason(
                previous_fast_ma=previous_fast,
                previous_slow_ma=previous_slow,
                current_fast_ma=current_fast,
                current_slow_ma=current_slow,
                current_price=current_price,
                entry_price=self.entry_price,
                holding_bars=holding_bars,
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                max_holding_bars=self.max_holding_bars,
            )
            if reason:
                self.position.close()
                self.entry_price = None
                self.entry_bar = None
            return

        if should_enter_ma_trend(
            previous_fast_ma=previous_fast,
            previous_slow_ma=previous_slow,
            current_fast_ma=current_fast,
            current_slow_ma=current_slow,
            close=current_price,
            trend_ma_value=trend_value,
            momentum_return=momentum_return,
        ):
            self.buy(size=self.position_pct)
            self.entry_price = current_price
            self.entry_bar = len(self.data.Close)
