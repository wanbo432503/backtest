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


class MA60PriceCrossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_pct: float = Field(default=0.95, ge=0.05, le=1)


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
    frame["ma_value"] = frame["Close"].astype(float).rolling(MA_PERIOD).mean()
    return frame


def evaluate_ma60_price_cross(context: StrategyBarContext) -> StrategyDecision:
    config = MA60PriceCrossConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < ma60_price_cross_min_history_bars(config):
        return StrategyDecision()
    previous = context.previous
    if previous is None:
        return StrategyDecision()
    current = context.current
    values = (
        previous["Close"],
        previous["ma_value"],
        current["Close"],
        current["ma_value"],
    )
    if any(pd.isna(value) for value in values):
        return StrategyDecision()

    previous_close, previous_ma, current_close, current_ma = map(float, values)
    if context.position is not None:
        if crossed_below_ma(previous_close, previous_ma, current_close, current_ma):
            return StrategyDecision(exit=ExitIntent("price_crossed_below_ma60"))
        return StrategyDecision()

    if not crossed_above_ma(previous_close, previous_ma, current_close, current_ma):
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
            metadata={"ma_cross_count": cross_count},
        )
    )


def ma60_price_cross_min_history_bars(config: BaseModel) -> int:
    MA60PriceCrossConfig.model_validate(config.model_dump())
    return MA_PERIOD + 1


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="ma60_price_cross",
    display_name="MA60价格穿越策略",
    description="收盘价上穿 MA60 后次日开盘买入，下穿 MA60 后次日开盘卖出；组合中优先选择历史交叉次数较少的股票。",
    config_model=MA60PriceCrossConfig,
    parameters=(
        StrategyParamMeta(
            name="position_pct",
            label="仓位比例",
            type="float",
            default=0.95,
            search_values=[0.5, 0.8, 0.95],
            min_value=0.05,
            max_value=1,
            step=0.05,
        ),
    ),
    prepare_frame=prepare_ma60_price_cross_frame,
    evaluate=evaluate_ma60_price_cross,
    min_history_bars=ma60_price_cross_min_history_bars,
    portfolio_priority_history_bars=PORTFOLIO_PRIORITY_HISTORY_BARS,
)
