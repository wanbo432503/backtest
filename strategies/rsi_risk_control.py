import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from strategy_engine import (
    EntryIntent,
    ExitIntent,
    StrategyBarContext,
    StrategyDecision,
    StrategyDefinition,
    StrategyParamMeta,
)


class RSIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rsi_period: int = Field(default=14, ge=2, le=60)
    rsi_buy: int = Field(default=30, ge=1, le=50)
    rsi_sell: int = Field(default=70, ge=50, le=99)
    trend_ma: int = Field(default=60, ge=5, le=250)
    stop_loss_pct: float = Field(default=5, ge=0, le=30)
    take_profit_pct: float = Field(default=12, ge=0, le=80)
    max_holding_bars: int = Field(default=120, ge=1, le=500)
    position_pct: float = Field(default=0.95, ge=0.05, le=1)
    cooldown_bars: int = Field(default=3, ge=0, le=60)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "RSIConfig":
        if self.rsi_buy >= self.rsi_sell:
            raise ValueError("rsi_buy must be below rsi_sell")
        return self


def calculate_rsi(values, period):
    prices = pd.Series(values, dtype="float64")
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).to_numpy()


def crossed_above(previous_value, current_value, threshold) -> bool:
    return previous_value <= threshold < current_value


def should_enter_long(
    previous_rsi,
    current_rsi,
    close,
    trend_ma_value,
    cooldown_remaining,
    rsi_buy,
) -> bool:
    if cooldown_remaining > 0:
        return False
    if np.isnan(trend_ma_value) or close <= trend_ma_value:
        return False
    return crossed_above(previous_rsi, current_rsi, rsi_buy)


def get_exit_reason(
    current_price,
    entry_price,
    previous_rsi,
    current_rsi,
    close,
    trend_ma_value,
    holding_bars,
    rsi_sell,
    stop_loss_pct,
    take_profit_pct,
    max_holding_bars,
):
    if entry_price is None:
        return None
    if stop_loss_pct and current_price <= entry_price * (1 - stop_loss_pct / 100):
        return "stop_loss"
    if take_profit_pct and current_price >= entry_price * (1 + take_profit_pct / 100):
        return "take_profit"
    if max_holding_bars and holding_bars >= max_holding_bars:
        return "max_holding_bars"
    if crossed_above(previous_rsi, current_rsi, rsi_sell):
        return "rsi_sell"
    if not np.isnan(trend_ma_value) and close < trend_ma_value:
        return "trend_break"
    return None


def prepare_rsi_frame(data: pd.DataFrame, config: BaseModel) -> pd.DataFrame:
    resolved = RSIConfig.model_validate(config.model_dump())
    frame = data.copy()
    close = frame["Close"].astype(float)
    frame["rsi"] = calculate_rsi(close.to_numpy(), resolved.rsi_period)
    frame["trend_ma_value"] = close.rolling(resolved.trend_ma).mean()
    return frame


def evaluate_rsi(context: StrategyBarContext) -> StrategyDecision:
    config = RSIConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < rsi_min_history_bars(config):
        return StrategyDecision()

    current = context.current
    previous = context.previous
    if previous is None:
        return StrategyDecision()

    state = dict(context.state)
    current_price = float(current["Close"])
    previous_rsi = float(previous["rsi"])
    current_rsi = float(current["rsi"])
    trend_value = float(current["trend_ma_value"])

    if context.position is not None:
        reason = get_exit_reason(
            current_price=current_price,
            entry_price=context.position.entry_price,
            previous_rsi=previous_rsi,
            current_rsi=current_rsi,
            close=current_price,
            trend_ma_value=trend_value,
            holding_bars=context.position.holding_bars,
            rsi_sell=config.rsi_sell,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            max_holding_bars=config.max_holding_bars,
        )
        if reason:
            return StrategyDecision(
                exit=ExitIntent(reason=reason),
                next_state={"cooldown_remaining": config.cooldown_bars},
            )
        return StrategyDecision(next_state=state)

    cooldown_remaining = int(state.get("cooldown_remaining", 0))
    if cooldown_remaining > 0:
        return StrategyDecision(
            next_state={"cooldown_remaining": cooldown_remaining - 1}
        )

    if should_enter_long(
        previous_rsi=previous_rsi,
        current_rsi=current_rsi,
        close=current_price,
        trend_ma_value=trend_value,
        cooldown_remaining=0,
        rsi_buy=config.rsi_buy,
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
            next_state={"cooldown_remaining": 0},
        )
    return StrategyDecision(next_state={"cooldown_remaining": 0})


def rsi_min_history_bars(config: BaseModel) -> int:
    resolved = RSIConfig.model_validate(config.model_dump())
    return max(resolved.rsi_period + 2, resolved.trend_ma + 1)


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="rsi_risk_control",
    display_name="RSI风控策略",
    description="RSI 上穿买入阈值且处于趋势均线上方时买入，并使用止损、止盈、持仓周期和冷却期控制风险。",
    config_model=RSIConfig,
    parameters=(
        StrategyParamMeta(name="rsi_period", label="RSI周期", type="int", default=14, search_values=[6, 14, 21], min_value=2, max_value=60, step=1),
        StrategyParamMeta(name="rsi_buy", label="买入阈值", type="int", default=30, search_values=[25, 30, 35], min_value=1, max_value=50, step=1),
        StrategyParamMeta(name="rsi_sell", label="卖出阈值", type="int", default=70, search_values=[60, 70, 80], min_value=50, max_value=99, step=1),
        StrategyParamMeta(name="trend_ma", label="趋势均线", type="int", default=60, search_values=[30, 60, 120], min_value=5, max_value=250, step=1),
        StrategyParamMeta(name="stop_loss_pct", label="止损比例", type="float", default=5, search_values=[3, 5, 8], min_value=0, max_value=30, step=0.5),
        StrategyParamMeta(name="take_profit_pct", label="止盈比例", type="float", default=12, search_values=[8, 12, 20], min_value=0, max_value=80, step=0.5),
        StrategyParamMeta(name="max_holding_bars", label="最大持仓周期", type="int", default=120, search_values=[40, 80, 120], min_value=1, max_value=500, step=1),
        StrategyParamMeta(name="position_pct", label="仓位比例", type="float", default=0.95, search_values=[0.5, 0.8, 0.95], min_value=0.05, max_value=1, step=0.05),
        StrategyParamMeta(name="cooldown_bars", label="冷却周期", type="int", default=3, search_values=[0, 3, 5], min_value=0, max_value=60, step=1),
    ),
    prepare_frame=prepare_rsi_frame,
    evaluate=evaluate_rsi,
    min_history_bars=rsi_min_history_bars,
)
