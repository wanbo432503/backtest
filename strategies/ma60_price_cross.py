from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from strategy_engine import (
    EntryIntent,
    ExitIntent,
    StrategyBarContext,
    StrategyDecision,
    StrategyDefinition,
    StrategyParamMeta,
)


MA_PERIOD = 60
CROSSING_LOOKBACK_BARS = 250
PORTFOLIO_PRIORITY_HISTORY_BARS = MA_PERIOD + CROSSING_LOOKBACK_BARS
ATR_PERIOD = 14
MA_SLOPE_LOOKBACK_BARS = 20
ENTRY_ATR_MULTIPLIER = 0.5
EXIT_ATR_MULTIPLIER = 0.25
MIN_MA_SLOPE_RETURN = 0.01
REENTRY_COOLDOWN_BARS = 10


class MA60PriceCrossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_pct: float = Field(default=0.15, ge=0.05, le=0.15)


def crossed_above_ma(
    previous_close: float,
    previous_ma: float,
    current_close: float,
    current_ma: float,
) -> bool:
    return previous_close <= previous_ma and current_close > current_ma


def crossed_below_ma(
    previous_close: float,
    previous_ma: float,
    current_close: float,
    current_ma: float,
) -> bool:
    return previous_close >= previous_ma and current_close < current_ma


def count_ma_crosses(
    close: pd.Series,
    moving_average: pd.Series,
    *,
    lookback_bars: int = CROSSING_LOOKBACK_BARS,
) -> int:
    previous_close = close.shift(1)
    previous_ma = moving_average.shift(1)
    valid = (
        close.notna()
        & moving_average.notna()
        & previous_close.notna()
        & previous_ma.notna()
    )
    crossed_up = (previous_close <= previous_ma) & (close > moving_average)
    crossed_down = (previous_close >= previous_ma) & (close < moving_average)
    return int(((crossed_up | crossed_down) & valid).tail(lookback_bars).sum())


def prepare_ma60_price_cross_frame(
    data: pd.DataFrame,
    config: BaseModel,
) -> pd.DataFrame:
    MA60PriceCrossConfig.model_validate(config.model_dump())
    frame = data.copy()
    close = frame["Close"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    frame["ma_value"] = close.rolling(MA_PERIOD).mean()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_value"] = true_range.rolling(ATR_PERIOD).mean()
    frame["ma_slope_return"] = (
        frame["ma_value"] / frame["ma_value"].shift(MA_SLOPE_LOOKBACK_BARS) - 1
    )
    return frame


def evaluate_ma60_price_cross(context: StrategyBarContext) -> StrategyDecision:
    config = MA60PriceCrossConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < ma60_price_cross_min_history_bars(config):
        return StrategyDecision()
    previous = context.previous
    if previous is None or context.bar_index < 2:
        return StrategyDecision()
    before_previous = context.frame.iloc[context.bar_index - 2]
    current = context.current
    values = (
        before_previous["Close"],
        before_previous["ma_value"],
        previous["Close"],
        previous["ma_value"],
        previous["atr_value"],
        current["Close"],
        current["ma_value"],
        current["atr_value"],
        current["ma_slope_return"],
    )
    if any(pd.isna(value) for value in values):
        return StrategyDecision()

    (
        before_previous_close,
        before_previous_ma,
        previous_close,
        previous_ma,
        previous_atr,
        current_close,
        current_ma,
        current_atr,
        ma_slope_return,
    ) = map(float, values)
    entry_threshold = current_ma + ENTRY_ATR_MULTIPLIER * current_atr
    exit_threshold = current_ma - EXIT_ATR_MULTIPLIER * current_atr
    if context.position is not None:
        if current_close < exit_threshold:
            return StrategyDecision(exit=ExitIntent("price_below_ma60_atr_band"))
        return StrategyDecision()

    if (
        context.bars_since_exit is not None
        and context.bars_since_exit <= REENTRY_COOLDOWN_BARS
    ):
        return StrategyDecision()
    if ma_slope_return < MIN_MA_SLOPE_RETURN:
        return StrategyDecision()
    if previous_close <= previous_ma or current_close <= entry_threshold:
        return StrategyDecision()

    previous_entry_threshold = (
        previous_ma + ENTRY_ATR_MULTIPLIER * previous_atr
    )
    crossed_upper_band = (
        previous_close <= previous_entry_threshold
        and current_close > entry_threshold
    )
    confirmed_ma_cross = (
        before_previous_close <= before_previous_ma
        and previous_close > previous_ma
        and current_close > entry_threshold
    )
    if context.bars_since_exit is not None:
        if not crossed_upper_band:
            return StrategyDecision()
    elif not (crossed_upper_band or confirmed_ma_cross):
        return StrategyDecision()
    cross_count = count_ma_crosses(
        context.history["Close"].astype(float),
        context.history["ma_value"].astype(float),
    )
    return StrategyDecision(
        entry=EntryIntent(
            order_type="next_open",
            strength=-float(cross_count),
            suggested_position_pct=config.position_pct,
            metadata={
                "ma_cross_count": cross_count,
                "entry_threshold": round(entry_threshold, 6),
                "ma_slope_return": round(ma_slope_return, 8),
            },
        )
    )


def ma60_price_cross_min_history_bars(config: BaseModel) -> int:
    MA60PriceCrossConfig.model_validate(config.model_dump())
    return MA_PERIOD + MA_SLOPE_LOOKBACK_BARS


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="ma60_price_cross",
    display_name="MA60价格穿越策略",
    description="连续两日站上 MA60、突破 0.5 ATR 上轨且 MA60 向上时次日开盘买入；跌破 0.25 ATR 下轨时卖出，卖出后冷却 10 日；组合中优先选择历史交叉次数较少的股票。",
    config_model=MA60PriceCrossConfig,
    parameters=(
        StrategyParamMeta(
            name="position_pct",
            label="仓位比例",
            type="float",
            default=0.15,
            search_values=[0.1, 0.15],
            min_value=0.05,
            max_value=0.15,
            step=0.05,
        ),
    ),
    prepare_frame=prepare_ma60_price_cross_frame,
    evaluate=evaluate_ma60_price_cross,
    min_history_bars=ma60_price_cross_min_history_bars,
    portfolio_priority_history_bars=PORTFOLIO_PRIORITY_HISTORY_BARS,
)
