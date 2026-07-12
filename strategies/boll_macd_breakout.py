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
from strategies.macd_volume_divergence_risk_control import macd_dea, macd_dif


class BollMACDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boll_period: int = Field(default=20, ge=5, le=120)
    boll_stddev: float = Field(default=2.0, ge=0.5, le=5)
    fast_period: int = Field(default=12, ge=2, le=60)
    slow_period: int = Field(default=26, ge=5, le=120)
    signal_period: int = Field(default=9, ge=2, le=40)
    macd_confirmation_bars: int = Field(default=5, ge=1, le=30)
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 1.0
    position_pct: float = Field(default=0.95, ge=0.05, le=1)

    @model_validator(mode="after")
    def validate_parameters(self) -> "BollMACDConfig":
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be below slow_period")
        validate_boll_macd_risk_percentages(self.stop_loss_pct, self.take_profit_pct)
        return self


def bollinger_middle(values, period):
    prices = pd.Series(values, dtype="float64")
    return prices.rolling(period).mean().to_numpy()


def bollinger_upper(values, period, stddev):
    prices = pd.Series(values, dtype="float64")
    rolling = prices.rolling(period)
    return (rolling.mean() + rolling.std(ddof=0) * stddev).to_numpy()


def should_enter_boll_macd_breakout(
    previous_close,
    previous_middle,
    previous_upper,
    current_close,
    current_middle,
    current_upper,
    current_dif,
    current_dea,
    recent_macd_golden_cross,
) -> bool:
    values = [
        previous_close,
        previous_middle,
        previous_upper,
        current_close,
        current_middle,
        current_upper,
        current_dif,
        current_dea,
    ]
    if any(np.isnan(float(value)) for value in values):
        return False

    middle_is_rising = current_middle > previous_middle
    crossed_upper_band = previous_close <= previous_upper and current_close > current_upper
    macd_is_bullish = current_dif > current_dea
    return (
        middle_is_rising
        and crossed_upper_band
        and macd_is_bullish
        and bool(recent_macd_golden_cross)
    )


def has_recent_macd_golden_cross(dif_values, dea_values, confirmation_bars) -> bool:
    confirmation_bars = int(confirmation_bars)
    if confirmation_bars < 1:
        raise ValueError("macd_confirmation_bars must be at least 1")

    dif = np.asarray(dif_values, dtype="float64")
    dea = np.asarray(dea_values, dtype="float64")
    if len(dif) != len(dea) or len(dif) < 2:
        return False

    transition_count = min(confirmation_bars, len(dif) - 1)
    start = len(dif) - transition_count
    for current_index in range(start, len(dif)):
        previous_index = current_index - 1
        values = [
            dif[previous_index],
            dea[previous_index],
            dif[current_index],
            dea[current_index],
        ]
        if any(np.isnan(value) for value in values):
            continue
        if dif[previous_index] <= dea[previous_index] and dif[current_index] > dea[current_index]:
            return True
    return False


def validate_boll_macd_risk_percentages(stop_loss_pct, take_profit_pct):
    stop_loss_pct = float(stop_loss_pct)
    take_profit_pct = float(take_profit_pct)
    risk_values = [stop_loss_pct, take_profit_pct]
    valid_values = all(
        np.isfinite(value)
        and 0.1 <= value <= 10
        and np.isclose(value * 10, round(value * 10), rtol=0, atol=1e-9)
        for value in risk_values
    )
    if not valid_values:
        raise ValueError(
            "stop_loss_pct and take_profit_pct must be between 0.1 and 10.0 "
            "in 0.1 increments"
        )
    return stop_loss_pct, take_profit_pct


def get_boll_macd_risk_prices(entry_price, stop_loss_pct, take_profit_pct):
    stop_loss_pct, take_profit_pct = validate_boll_macd_risk_percentages(
        stop_loss_pct,
        take_profit_pct,
    )
    stop_price = entry_price * (1 - stop_loss_pct / 100)
    take_price = entry_price * (1 + take_profit_pct / 100)
    return stop_price, take_price


def prepare_boll_macd_frame(data: pd.DataFrame, config: BaseModel) -> pd.DataFrame:
    resolved = BollMACDConfig.model_validate(config.model_dump())
    frame = data.copy()
    close = frame["Close"].astype(float).to_numpy()
    frame["boll_middle"] = bollinger_middle(close, resolved.boll_period)
    frame["boll_upper"] = bollinger_upper(close, resolved.boll_period, resolved.boll_stddev)
    frame["macd_dif"] = macd_dif(close, resolved.fast_period, resolved.slow_period, resolved.signal_period)
    frame["macd_dea"] = macd_dea(close, resolved.fast_period, resolved.slow_period, resolved.signal_period)
    return frame


def evaluate_boll_macd(context: StrategyBarContext) -> StrategyDecision:
    config = BollMACDConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < boll_macd_min_history_bars(config):
        return StrategyDecision()
    previous = context.previous
    if previous is None:
        return StrategyDecision()
    current = context.current
    current_price = float(current["Close"])
    if context.position is not None:
        stop_price, target_price = get_boll_macd_risk_prices(
            context.position.entry_price,
            config.stop_loss_pct,
            config.take_profit_pct,
        )
        reason = None
        if current_price <= stop_price:
            reason = "stop_loss"
        elif current_price >= target_price:
            reason = "take_profit"
        return StrategyDecision(exit=ExitIntent(reason) if reason else None)

    history = context.history
    confirmation = config.macd_confirmation_bars
    if should_enter_boll_macd_breakout(
        previous_close=float(previous["Close"]),
        previous_middle=float(previous["boll_middle"]),
        previous_upper=float(previous["boll_upper"]),
        current_close=current_price,
        current_middle=float(current["boll_middle"]),
        current_upper=float(current["boll_upper"]),
        current_dif=float(current["macd_dif"]),
        current_dea=float(current["macd_dea"]),
        recent_macd_golden_cross=has_recent_macd_golden_cross(
            history["macd_dif"].iloc[-(confirmation + 1) :].to_numpy(),
            history["macd_dea"].iloc[-(confirmation + 1) :].to_numpy(),
            confirmation,
        ),
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


def boll_macd_min_history_bars(config: BaseModel) -> int:
    resolved = BollMACDConfig.model_validate(config.model_dump())
    return max(resolved.boll_period, resolved.slow_period + resolved.signal_period) + 2


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="boll_macd_breakout",
    display_name="BOLL+MACD上轨突破策略",
    description="布林中轨向上、收盘价上穿上轨、MACD保持多头且近期发生金叉时买入，按可优化比例止盈止损。",
    config_model=BollMACDConfig,
    parameters=(
        StrategyParamMeta(name="boll_period", label="布林周期", type="int", default=20, search_values=[20], min_value=5, max_value=120, step=1),
        StrategyParamMeta(name="boll_stddev", label="布林标准差倍数", type="float", default=2.0, search_values=[2.0], min_value=0.5, max_value=5, step=0.1),
        StrategyParamMeta(name="fast_period", label="MACD快线周期", type="int", default=12, search_values=[12], min_value=2, max_value=60, step=1),
        StrategyParamMeta(name="slow_period", label="MACD慢线周期", type="int", default=26, search_values=[26], min_value=5, max_value=120, step=1),
        StrategyParamMeta(name="signal_period", label="DEA平滑周期", type="int", default=9, search_values=[9], min_value=2, max_value=40, step=1),
        StrategyParamMeta(name="macd_confirmation_bars", label="MACD金叉确认窗口", type="int", default=5, search_values=[3, 5, 10], min_value=1, max_value=30, step=1),
        StrategyParamMeta(name="stop_loss_pct", label="止损比例", type="float", default=1.0, search_values=[0.5, 1.0, 1.5, 2.0, 3.0], min_value=0.1, max_value=10, step=0.1),
        StrategyParamMeta(name="take_profit_pct", label="止盈比例", type="float", default=1.0, search_values=[0.5, 1.0, 1.5, 2.0, 3.0], min_value=0.1, max_value=10, step=0.1),
        StrategyParamMeta(name="position_pct", label="仓位比例", type="float", default=0.95, search_values=[0.95], min_value=0.05, max_value=1, step=0.05),
    ),
    prepare_frame=prepare_boll_macd_frame,
    evaluate=evaluate_boll_macd,
    min_history_bars=boll_macd_min_history_bars,
)
