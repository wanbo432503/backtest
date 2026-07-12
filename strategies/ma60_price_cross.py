from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from strategy_engine import (
    EntryIntent,
    RiskIntent,
    StrategyBarContext,
    StrategyDecision,
    StrategyDefinition,
    StrategyParamMeta,
)


MA_PERIOD = 60
CROSSING_LOOKBACK_BARS = 250
PORTFOLIO_PRIORITY_HISTORY_BARS = CROSSING_LOOKBACK_BARS
PORTFOLIO_INDICATOR_WARMUP_BARS = MA_PERIOD
ATR_PERIOD = 14
ENTRY_ATR_MULTIPLIER = 0.5
EXIT_ATR_MULTIPLIER = 0.25
REENTRY_COOLDOWN_BARS = 10


class MA60PriceCrossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_pct: float = Field(default=0.15, ge=0.05, le=0.15)
    ma_slope_lookback_bars: int = Field(default=20, ge=5, le=60)
    min_ma_slope_return_pct: float = Field(default=1.0, ge=-5, le=5)
    max_entry_gap_pct: float = Field(default=3.0, ge=0, le=20)


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
    validated = MA60PriceCrossConfig.model_validate(config.model_dump())
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
        frame["ma_value"]
        / frame["ma_value"].shift(validated.ma_slope_lookback_bars)
        - 1
    )
    return frame


def evaluate_ma60_price_cross(context: StrategyBarContext) -> StrategyDecision:
    config = MA60PriceCrossConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < ma60_price_cross_min_history_bars(config):
        return StrategyDecision()
    current = context.current
    values = (
        current["Close"],
        current["ma_value"],
        current["atr_value"],
        current["ma_slope_return"],
    )
    if any(pd.isna(value) for value in values):
        return StrategyDecision()

    (
        current_close,
        current_ma,
        current_atr,
        ma_slope_return,
    ) = map(float, values)
    entry_threshold = current_ma + ENTRY_ATR_MULTIPLIER * current_atr
    exit_threshold = current_ma - EXIT_ATR_MULTIPLIER * current_atr
    if context.position is not None:
        return StrategyDecision(
            risk_update=RiskIntent(
                stop_price=round(exit_threshold, 6),
                stop_reason="price_below_ma60_atr_band",
            )
        )

    if (
        context.bars_since_exit is not None
        and context.bars_since_exit <= REENTRY_COOLDOWN_BARS
    ):
        return StrategyDecision()
    is_breakout_setup = current_close <= entry_threshold
    minimum_slope_return = config.min_ma_slope_return_pct / 100
    if not is_breakout_setup and ma_slope_return < minimum_slope_return:
        return StrategyDecision()
    entry_mode = (
        "breakout"
        if is_breakout_setup
        else "trend_continuation"
    )
    cross_count = count_ma_crosses(
        context.entry_history["Close"].astype(float),
        context.entry_history["ma_value"].astype(float),
    )
    return StrategyDecision(
        entry=EntryIntent(
            order_type="stop_next_bar",
            strength=-float(cross_count),
            trigger_price=round(entry_threshold, 6),
            expires_after_bars=1,
            suggested_position_pct=config.position_pct,
            risk=RiskIntent(
                stop_price=round(exit_threshold, 6),
                stop_reason="price_below_ma60_atr_band",
            ),
            metadata={
                "ma_cross_count": cross_count,
                "entry_threshold": round(entry_threshold, 6),
                "ma_slope_return": round(ma_slope_return, 8),
                "entry_mode": entry_mode,
                "max_entry_gap_pct": config.max_entry_gap_pct,
            },
        )
    )


def ma60_price_cross_min_history_bars(config: BaseModel) -> int:
    validated = MA60PriceCrossConfig.model_validate(config.model_dump())
    return MA_PERIOD + validated.ma_slope_lookback_bars


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="ma60_price_cross",
    display_name="MA60价格穿越策略",
    description="每日收盘后按当日 MA60 与 ATR 预挂次日 0.5 ATR 上轨突破单，首次突破不受斜率限制；已经站上上轨的趋势延续入场需满足可配置的 MA60 斜率，并限制跳空追价；持仓后使用 0.25 ATR 下轨保护，卖出后冷却 10 日；组合回测开始后的前 250 根 K 线只观察不交易。",
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
        StrategyParamMeta(
            name="ma_slope_lookback_bars",
            label="MA60斜率回看周期",
            type="int",
            default=20,
            search_values=[10, 20, 30, 40],
            min_value=5,
            max_value=60,
            step=5,
            description="仅用于趋势延续入场，计算 MA60 在指定交易日内的变化。",
        ),
        StrategyParamMeta(
            name="min_ma_slope_return_pct",
            label="最低MA60斜率收益(%)",
            type="float",
            default=1.0,
            search_values=[0, 0.5, 1, 2],
            min_value=-5,
            max_value=5,
            step=0.5,
            description="仅限制已经站上 ATR 上轨后的趋势延续入场。",
        ),
        StrategyParamMeta(
            name="max_entry_gap_pct",
            label="最大追价幅度(%)",
            type="float",
            default=3.0,
            search_values=[1, 2, 3, 5],
            min_value=0,
            max_value=20,
            step=0.5,
            description="次日开盘高出 ATR 触发价超过该比例时取消买入。",
        ),
    ),
    prepare_frame=prepare_ma60_price_cross_frame,
    evaluate=evaluate_ma60_price_cross,
    min_history_bars=ma60_price_cross_min_history_bars,
    portfolio_priority_history_bars=PORTFOLIO_PRIORITY_HISTORY_BARS,
    portfolio_indicator_warmup_bars=PORTFOLIO_INDICATOR_WARMUP_BARS,
)
