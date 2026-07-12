import numpy as np
import pandas as pd
from backtesting import Strategy
from pydantic import BaseModel, ConfigDict, Field

from strategy_engine import (
    EntryIntent,
    ExitIntent,
    StrategyBarContext,
    StrategyDecision,
    StrategyDefinition,
    StrategyParamMeta,
)


class VolumeBreakoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    breakout_lookback: int = Field(default=20, ge=3, le=120)
    volume_lookback: int = Field(default=10, ge=3, le=80)
    volume_multiplier: float = Field(default=2.0, ge=1, le=10)
    stop_loss_pct: float = Field(default=5, ge=0, le=30)
    take_profit_pct: float = Field(default=12, ge=0, le=80)
    max_holding_bars: int = Field(default=80, ge=1, le=500)
    position_pct: float = Field(default=0.95, ge=0.05, le=1)
    limit_up_down_filter: bool = True


def rolling_max(values, window):
    return pd.Series(values, dtype="float64").rolling(window).max().to_numpy()


def rolling_mean(values, window):
    return pd.Series(values, dtype="float64").rolling(window).mean().to_numpy()


def is_limit_up(close, previous_close, threshold=0.098) -> bool:
    if previous_close <= 0:
        return False
    return close / previous_close - 1 >= threshold


def should_enter_breakout(
    close,
    previous_highest_close,
    volume,
    average_volume,
    volume_multiplier,
    previous_close,
    limit_up_down_filter,
) -> bool:
    values = [close, previous_highest_close, volume, average_volume, previous_close]
    if any(np.isnan(float(value)) for value in values):
        return False
    if close <= previous_highest_close:
        return False
    if average_volume <= 0 or volume < average_volume * volume_multiplier:
        return False
    if limit_up_down_filter and is_limit_up(close, previous_close):
        return False
    return True


def get_breakout_exit_reason(
    current_price,
    breakout_line,
    entry_price,
    holding_bars,
    stop_loss_pct,
    take_profit_pct,
    max_holding_bars,
):
    if breakout_line is not None and current_price < breakout_line:
        return "breakout_line_lost"
    if entry_price is None:
        return None
    if stop_loss_pct and current_price <= entry_price * (1 - stop_loss_pct / 100):
        return "stop_loss"
    if take_profit_pct and current_price >= entry_price * (1 + take_profit_pct / 100):
        return "take_profit"
    if max_holding_bars and holding_bars >= max_holding_bars:
        return "max_holding_bars"
    return None


def prepare_volume_breakout_frame(data: pd.DataFrame, config: BaseModel) -> pd.DataFrame:
    resolved = VolumeBreakoutConfig.model_validate(config.model_dump())
    frame = data.copy()
    frame["highest_close"] = frame["Close"].astype(float).rolling(resolved.breakout_lookback).max()
    frame["average_volume"] = frame["Volume"].astype(float).rolling(resolved.volume_lookback).mean()
    return frame


def evaluate_volume_breakout(context: StrategyBarContext) -> StrategyDecision:
    config = VolumeBreakoutConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < volume_breakout_min_history_bars(config):
        return StrategyDecision()
    previous = context.previous
    if previous is None:
        return StrategyDecision()
    current = context.current
    state = dict(context.state)
    current_price = float(current["Close"])
    if context.position is not None:
        reason = get_breakout_exit_reason(
            current_price=current_price,
            breakout_line=state.get("breakout_line"),
            entry_price=context.position.entry_price,
            holding_bars=context.position.holding_bars,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            max_holding_bars=config.max_holding_bars,
        )
        return StrategyDecision(
            exit=ExitIntent(reason) if reason else None,
            next_state={} if reason else state,
        )
    breakout_line = float(previous["highest_close"])
    if should_enter_breakout(
        close=current_price,
        previous_highest_close=breakout_line,
        volume=float(current["Volume"]),
        average_volume=float(previous["average_volume"]),
        volume_multiplier=config.volume_multiplier,
        previous_close=float(previous["Close"]),
        limit_up_down_filter=config.limit_up_down_filter,
    ):
        return StrategyDecision(
            entry=EntryIntent(
                order_type="next_open",
                suggested_position_pct=config.position_pct,
                metadata={
                    "stop_loss_pct": config.stop_loss_pct,
                    "take_profit_pct": config.take_profit_pct,
                },
            ),
            next_state={"breakout_line": breakout_line},
        )
    return StrategyDecision(next_state=state)


def volume_breakout_min_history_bars(config: BaseModel) -> int:
    resolved = VolumeBreakoutConfig.model_validate(config.model_dump())
    return max(resolved.breakout_lookback, resolved.volume_lookback) + 2


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="volume_breakout_risk_control",
    display_name="放量突破风控策略",
    description="价格突破近期高点且成交量显著放大时买入，跌破突破线或触发风控时卖出。",
    config_model=VolumeBreakoutConfig,
    parameters=(
        StrategyParamMeta(name="breakout_lookback", label="突破窗口", type="int", default=20, search_values=[10, 20, 40], min_value=3, max_value=120, step=1),
        StrategyParamMeta(name="volume_lookback", label="均量窗口", type="int", default=10, search_values=[5, 10, 20], min_value=3, max_value=80, step=1),
        StrategyParamMeta(name="volume_multiplier", label="放量倍数", type="float", default=2.0, search_values=[1.5, 2.0, 3.0], min_value=1, max_value=10, step=0.1),
        StrategyParamMeta(name="stop_loss_pct", label="止损比例", type="float", default=5, search_values=[3, 5, 8], min_value=0, max_value=30, step=0.5),
        StrategyParamMeta(name="take_profit_pct", label="止盈比例", type="float", default=12, search_values=[8, 12, 20], min_value=0, max_value=80, step=0.5),
        StrategyParamMeta(name="max_holding_bars", label="最大持仓周期", type="int", default=80, search_values=[40, 80, 120], min_value=1, max_value=500, step=1),
        StrategyParamMeta(name="position_pct", label="仓位比例", type="float", default=0.95, search_values=[0.5, 0.8, 0.95], min_value=0.05, max_value=1, step=0.05),
        StrategyParamMeta(name="limit_up_down_filter", label="过滤涨停", type="bool", default=True, search_values=[True, False]),
    ),
    prepare_frame=prepare_volume_breakout_frame,
    evaluate=evaluate_volume_breakout,
    min_history_bars=volume_breakout_min_history_bars,
)


class VolumeBreakoutRiskControlStrategy(Strategy):
    """Volume-confirmed breakout strategy with fixed risk controls."""

    strategy_name = "volume_breakout_risk_control"
    display_name = "放量突破风控策略"
    description = "价格突破近期高点且成交量显著放大时买入，跌破突破线或触发风控时卖出。"

    breakout_lookback = 20
    volume_lookback = 10
    volume_multiplier = 2.0
    stop_loss_pct = 5
    take_profit_pct = 12
    max_holding_bars = 80
    position_pct = 0.95
    limit_up_down_filter = True

    def init(self):
        close = self.data.Close
        volume = self.data.Volume
        self.highest_close = self.I(rolling_max, close, self.breakout_lookback)
        self.average_volume = self.I(rolling_mean, volume, self.volume_lookback)
        self.entry_price = None
        self.entry_bar = None
        self.breakout_line = None

    def next(self):
        min_bars = max(self.breakout_lookback, self.volume_lookback) + 2
        if len(self.data.Close) < min_bars:
            return

        current_price = float(self.data.Close[-1])
        previous_close = float(self.data.Close[-2])
        previous_highest_close = float(self.highest_close[-2])
        current_volume = float(self.data.Volume[-1])
        average_volume = float(self.average_volume[-2])

        if self.position:
            holding_bars = len(self.data.Close) - (self.entry_bar or len(self.data.Close))
            reason = get_breakout_exit_reason(
                current_price=current_price,
                breakout_line=self.breakout_line,
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
                self.breakout_line = None
            return

        if should_enter_breakout(
            close=current_price,
            previous_highest_close=previous_highest_close,
            volume=current_volume,
            average_volume=average_volume,
            volume_multiplier=self.volume_multiplier,
            previous_close=previous_close,
            limit_up_down_filter=self.limit_up_down_filter,
        ):
            self.buy(size=self.position_pct)
            self.entry_price = current_price
            self.entry_bar = len(self.data.Close)
            self.breakout_line = previous_highest_close
