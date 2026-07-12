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


class MABreakoutATRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_ma: int = Field(default=20, ge=5, le=80)
    medium_ma: int = Field(default=60, ge=20, le=180)
    long_ma: int = Field(default=120, ge=60, le=250)
    breakout_lookback: int = Field(default=40, ge=5, le=180)
    volume_lookback: int = Field(default=20, ge=3, le=120)
    volume_multiplier: float = Field(default=1.5, ge=1, le=10)
    bootstrap_bars: int = Field(default=120, ge=0, le=250)
    atr_period: int = Field(default=14, ge=3, le=80)
    atr_stop_multiplier: float = Field(default=2.5, ge=0.5, le=10)
    max_holding_bars: int = Field(default=80, ge=1, le=500)
    target_atr_risk_pct: float = Field(default=0.02, ge=0.001, le=0.2)
    min_position_pct: float = Field(default=0.2, ge=0.05, le=1)
    max_position_pct: float = Field(default=0.95, ge=0.05, le=1)

    @model_validator(mode="after")
    def validate_periods_and_positions(self) -> "MABreakoutATRConfig":
        if not self.short_ma < self.medium_ma < self.long_ma:
            raise ValueError("moving-average periods must be strictly increasing")
        if self.min_position_pct > self.max_position_pct:
            raise ValueError("min_position_pct must not exceed max_position_pct")
        return self


def rolling_max(values, window):
    return pd.Series(values, dtype="float64").rolling(window).max().to_numpy()


def rolling_mean(values, window):
    return pd.Series(values, dtype="float64").rolling(window).mean().to_numpy()


def rolling_mean_min_periods(values, window, min_periods):
    return (
        pd.Series(values, dtype="float64")
        .rolling(window, min_periods=min(int(min_periods), int(window)))
        .mean()
        .to_numpy()
    )


def calculate_atr(high, low, close, period):
    high_series = pd.Series(high, dtype="float64")
    low_series = pd.Series(low, dtype="float64")
    close_series = pd.Series(close, dtype="float64")
    previous_close = close_series.shift(1)
    true_range = pd.concat(
        [
            high_series - low_series,
            (high_series - previous_close).abs(),
            (low_series - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean().to_numpy()


def should_enter_ma_breakout(
    close,
    ma20,
    ma60,
    ma120,
    previous_highest_high,
    volume,
    average_volume,
    volume_multiplier,
) -> bool:
    values = [close, ma20, ma60, ma120, previous_highest_high, volume, average_volume]
    if any(np.isnan(float(value)) for value in values):
        return False
    if close <= ma120:
        return False
    if ma20 <= ma60:
        return False
    if close <= previous_highest_high:
        return False
    if average_volume <= 0 or volume <= average_volume * volume_multiplier:
        return False
    return True


def should_enter_trend_bootstrap(
    close,
    ma20,
    ma60,
    previous_highest_high,
    volume,
    average_volume,
    volume_multiplier,
) -> bool:
    values = [close, ma20, ma60, previous_highest_high, volume, average_volume]
    if any(np.isnan(float(value)) for value in values):
        return False
    if close <= ma60:
        return False
    if ma20 <= ma60:
        return False
    if close <= previous_highest_high:
        return False
    if average_volume <= 0 or volume <= average_volume * volume_multiplier:
        return False
    return True


def is_weak_trend(close, ma20, ma60) -> bool:
    values = [close, ma20, ma60]
    if any(np.isnan(float(value)) for value in values):
        return False
    return ma20 <= ma60 or close < ma60


def get_ma_breakout_atr_exit_reason(
    close,
    ma20,
    ma60,
    highest_close,
    atr,
    atr_stop_multiplier,
    holding_bars,
    max_holding_bars,
):
    if np.isnan(float(close)) or np.isnan(float(ma20)):
        return None
    if close < ma20:
        return "ma20_lost"
    if (
        highest_close is not None
        and not np.isnan(float(highest_close))
        and not np.isnan(float(atr))
        and close < highest_close - atr_stop_multiplier * atr
    ):
        return "atr_trailing_stop"
    if holding_bars > max_holding_bars and is_weak_trend(close, ma20, ma60):
        return "late_weak_trend"
    return None


def calculate_atr_position_pct(
    close,
    atr,
    target_atr_risk_pct,
    min_position_pct,
    max_position_pct,
) -> float:
    if close <= 0 or atr <= 0 or np.isnan(float(close)) or np.isnan(float(atr)):
        return float(min_position_pct)
    atr_pct = atr / close
    if atr_pct <= 0:
        return float(max_position_pct)
    raw_position = target_atr_risk_pct / atr_pct
    clipped = min(max(raw_position, min_position_pct), max_position_pct)
    return round(float(clipped), 4)


def prepare_ma_breakout_atr_frame(data: pd.DataFrame, config: BaseModel) -> pd.DataFrame:
    resolved = MABreakoutATRConfig.model_validate(config.model_dump())
    frame = data.copy()
    close = frame["Close"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    volume = frame["Volume"].astype(float)
    frame["ma_short_value"] = close.rolling(resolved.short_ma).mean()
    frame["ma_medium_value"] = close.rolling(resolved.medium_ma).mean()
    frame["ma_long_value"] = close.rolling(
        resolved.long_ma,
        min_periods=min(resolved.medium_ma, resolved.long_ma),
    ).mean()
    frame["highest_high"] = high.rolling(resolved.breakout_lookback).max()
    frame["average_volume"] = volume.rolling(resolved.volume_lookback).mean()
    frame["atr"] = calculate_atr(high, low, close, resolved.atr_period)
    return frame


def evaluate_ma_breakout_atr(context: StrategyBarContext) -> StrategyDecision:
    config = MABreakoutATRConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < ma_breakout_atr_min_history_bars(config):
        return StrategyDecision()
    previous = context.previous
    if previous is None:
        return StrategyDecision()
    current = context.current
    current_close = float(current["Close"])
    current_atr = float(current["atr"])
    if context.position is not None:
        highest_close = max(
            context.position.highest_price or current_close,
            current_close,
        )
        reason = get_ma_breakout_atr_exit_reason(
            close=current_close,
            ma20=float(current["ma_short_value"]),
            ma60=float(current["ma_medium_value"]),
            highest_close=highest_close,
            atr=current_atr,
            atr_stop_multiplier=config.atr_stop_multiplier,
            holding_bars=context.position.holding_bars,
            max_holding_bars=config.max_holding_bars,
        )
        return StrategyDecision(
            exit=ExitIntent(reason) if reason else None,
            next_state={"highest_price": highest_close},
        )

    strict_min_bars = max(
        config.long_ma,
        config.breakout_lookback,
        config.volume_lookback,
        config.atr_period,
    ) + 2
    common = {
        "close": current_close,
        "ma20": float(current["ma_short_value"]),
        "ma60": float(current["ma_medium_value"]),
        "previous_highest_high": float(previous["highest_high"]),
        "volume": float(current["Volume"]),
        "average_volume": float(previous["average_volume"]),
        "volume_multiplier": config.volume_multiplier,
    }
    strict_entry = (
        context.bar_index + 1 >= strict_min_bars
        and should_enter_ma_breakout(
            **common,
            ma120=float(current["ma_long_value"]),
        )
    )
    bootstrap_entry = (
        not strict_entry
        and config.bootstrap_bars > 0
        and context.bar_index + 1 < strict_min_bars + config.bootstrap_bars
        and should_enter_trend_bootstrap(**common)
    )
    if strict_entry or bootstrap_entry:
        position_pct = calculate_atr_position_pct(
            close=current_close,
            atr=current_atr,
            target_atr_risk_pct=config.target_atr_risk_pct,
            min_position_pct=config.min_position_pct,
            max_position_pct=config.max_position_pct,
        )
        return StrategyDecision(
            entry=EntryIntent(
                order_type="next_open",
                suggested_position_pct=position_pct,
            ),
            next_state={"highest_price": current_close},
        )
    return StrategyDecision()


def ma_breakout_atr_min_history_bars(config: BaseModel) -> int:
    resolved = MABreakoutATRConfig.model_validate(config.model_dump())
    if resolved.bootstrap_bars > 0:
        return max(
            resolved.medium_ma,
            resolved.breakout_lookback,
            resolved.volume_lookback,
            resolved.atr_period,
        ) + 2
    return max(
        resolved.long_ma,
        resolved.breakout_lookback,
        resolved.volume_lookback,
        resolved.atr_period,
    ) + 2


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="ma_breakout_atr_risk_control",
    display_name="均线突破ATR风控策略",
    description="长期均线多头过滤、MA20/MA60 趋势确认、高点放量突破买入；回测开头可用启动捕捉窗口减少长均线预热漏判，并使用 ATR 移动止损和波动率仓位控制。",
    config_model=MABreakoutATRConfig,
    parameters=(
        StrategyParamMeta(name="short_ma", label="短期均线", type="int", default=20, search_values=[10, 20, 30], min_value=5, max_value=80, step=1),
        StrategyParamMeta(name="medium_ma", label="中期均线", type="int", default=60, search_values=[40, 60, 90], min_value=20, max_value=180, step=1),
        StrategyParamMeta(name="long_ma", label="长期均线", type="int", default=120, search_values=[90, 120, 200], min_value=60, max_value=250, step=1),
        StrategyParamMeta(name="breakout_lookback", label="突破窗口", type="int", default=40, search_values=[20, 40, 60], min_value=5, max_value=180, step=1),
        StrategyParamMeta(name="volume_lookback", label="均量窗口", type="int", default=20, search_values=[10, 20, 30], min_value=3, max_value=120, step=1),
        StrategyParamMeta(name="volume_multiplier", label="放量倍数", type="float", default=1.5, search_values=[1.2, 1.5, 2.0], min_value=1, max_value=10, step=0.1),
        StrategyParamMeta(name="bootstrap_bars", label="启动捕捉窗口", type="int", default=120, search_values=[0, 60, 120], min_value=0, max_value=250, step=1),
        StrategyParamMeta(name="atr_period", label="ATR周期", type="int", default=14, search_values=[10, 14, 20], min_value=3, max_value=80, step=1),
        StrategyParamMeta(name="atr_stop_multiplier", label="ATR止损倍数", type="float", default=2.5, search_values=[2.0, 2.5, 3.0], min_value=0.5, max_value=10, step=0.1),
        StrategyParamMeta(name="max_holding_bars", label="最大持仓周期", type="int", default=80, search_values=[60, 80, 120], min_value=1, max_value=500, step=1),
        StrategyParamMeta(name="target_atr_risk_pct", label="目标ATR风险", type="float", default=0.02, search_values=[0.015, 0.02, 0.03], min_value=0.001, max_value=0.2, step=0.001),
        StrategyParamMeta(name="min_position_pct", label="最低仓位", type="float", default=0.2, search_values=[0.1, 0.2, 0.3], min_value=0.05, max_value=1, step=0.05),
        StrategyParamMeta(name="max_position_pct", label="最高仓位", type="float", default=0.95, search_values=[0.6, 0.8, 0.95], min_value=0.05, max_value=1, step=0.05),
    ),
    prepare_frame=prepare_ma_breakout_atr_frame,
    evaluate=evaluate_ma_breakout_atr,
    min_history_bars=ma_breakout_atr_min_history_bars,
)
