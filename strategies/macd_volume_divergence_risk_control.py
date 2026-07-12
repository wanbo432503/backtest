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


class MACDVolumeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fast_period: int = Field(default=12, ge=2, le=60)
    slow_period: int = Field(default=26, ge=5, le=120)
    signal_period: int = Field(default=9, ge=2, le=40)
    volume_lookback: int = Field(default=20, ge=3, le=120)
    volume_multiplier: float = Field(default=2.0, ge=1, le=10)
    continuation_volume_multiplier: float = Field(default=1.2, ge=0.8, le=5)
    continuation_pullback_pct: float = Field(default=8, ge=0, le=30)
    divergence_lookback: int = Field(default=60, ge=10, le=180)
    zero_axis_threshold: float = Field(default=0.03, ge=0.001, le=0.2)
    trend_ma: int = Field(default=60, ge=5, le=250)
    histogram_fade_bars: int = Field(default=3, ge=2, le=10)
    stop_loss_pct: float = Field(default=5, ge=0, le=30)
    take_profit_pct: float = Field(default=12, ge=0, le=80)
    trailing_stop_pct: float = Field(default=10, ge=0, le=40)
    max_holding_bars: int = Field(default=80, ge=1, le=500)
    position_pct: float = Field(default=0.95, ge=0.05, le=1)

    @model_validator(mode="after")
    def validate_periods(self) -> "MACDVolumeConfig":
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be below slow_period")
        return self


def calculate_macd(values, fast_period=12, slow_period=26, signal_period=9):
    prices = pd.Series(values, dtype="float64")
    fast_ema = prices.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    dif = fast_ema - slow_ema
    dea = dif.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = dif - dea
    return (
        dif.fillna(0).to_numpy(),
        dea.fillna(0).to_numpy(),
        histogram.fillna(0).to_numpy(),
    )


def macd_dif(values, fast_period, slow_period, signal_period):
    return calculate_macd(values, fast_period, slow_period, signal_period)[0]


def macd_dea(values, fast_period, slow_period, signal_period):
    return calculate_macd(values, fast_period, slow_period, signal_period)[1]


def macd_histogram(values, fast_period, slow_period, signal_period):
    return calculate_macd(values, fast_period, slow_period, signal_period)[2]


def rolling_mean(values, window):
    return pd.Series(values, dtype="float64").rolling(window).mean().to_numpy()


def is_golden_cross(previous_dif, previous_dea, current_dif, current_dea) -> bool:
    return previous_dif <= previous_dea and current_dif > current_dea


def is_dead_cross(previous_dif, previous_dea, current_dif, current_dea) -> bool:
    return previous_dif >= previous_dea and current_dif < current_dea


def has_volume_confirmation(volume, average_volume, multiplier) -> bool:
    values = [volume, average_volume, multiplier]
    if any(np.isnan(float(value)) for value in values):
        return False
    return average_volume > 0 and volume >= average_volume * multiplier


def has_bullish_macd_divergence(closes, dif_values, histogram_values) -> bool:
    closes = np.asarray(closes, dtype="float64")
    dif_values = np.asarray(dif_values, dtype="float64")
    histogram_values = np.asarray(histogram_values, dtype="float64")
    if len(closes) < 6 or len(dif_values) != len(closes) or len(histogram_values) != len(closes):
        return False

    midpoint = len(closes) // 2
    early_close = closes[:midpoint]
    recent_close = closes[midpoint:]
    early_dif = dif_values[:midpoint]
    recent_dif = dif_values[midpoint:]
    early_histogram = histogram_values[:midpoint]
    recent_histogram = histogram_values[midpoint:]

    series_to_check = [
        early_close,
        recent_close,
        early_dif,
        recent_dif,
        early_histogram,
        recent_histogram,
    ]
    if any(np.isnan(series).all() for series in series_to_check):
        return False

    price_made_lower_low = np.nanmin(recent_close) < np.nanmin(early_close)
    dif_made_higher_low = np.nanmin(recent_dif) > np.nanmin(early_dif)
    histogram_made_higher_low = np.nanmin(recent_histogram) > np.nanmin(early_histogram)
    indicator_made_higher_low = dif_made_higher_low or histogram_made_higher_low
    return bool(price_made_lower_low and indicator_made_higher_low)


def is_near_zero_axis(current_dif, current_dea, current_close, zero_axis_threshold) -> bool:
    if current_close <= 0:
        return False
    distance = max(abs(current_dif), abs(current_dea)) / current_close
    return distance <= zero_axis_threshold


def should_enter_macd_volume(
    previous_dif,
    previous_dea,
    current_dif,
    current_dea,
    current_close,
    volume,
    average_volume,
    volume_multiplier,
    divergence_detected,
    zero_axis_threshold,
) -> bool:
    values = [previous_dif, previous_dea, current_dif, current_dea, current_close]
    if any(np.isnan(float(value)) for value in values):
        return False
    if not is_golden_cross(previous_dif, previous_dea, current_dif, current_dea):
        return False
    if not has_volume_confirmation(volume, average_volume, volume_multiplier):
        return False
    if divergence_detected:
        return True
    if is_near_zero_axis(current_dif, current_dea, current_close, zero_axis_threshold):
        return True
    return current_dif > 0 and current_dea > 0


def should_enter_continuation(
    current_dif,
    current_dea,
    current_close,
    trend_ma_value,
    volume,
    average_volume,
    continuation_volume_multiplier,
    continuation_pullback_pct,
) -> bool:
    if not is_trend_intact(current_dif, current_dea, current_close, trend_ma_value):
        return False
    if not has_volume_confirmation(volume, average_volume, continuation_volume_multiplier):
        return False
    if np.isnan(float(trend_ma_value)) or trend_ma_value <= 0:
        return False
    return current_close <= trend_ma_value * (1 + continuation_pullback_pct / 100)


def is_histogram_fading(recent_histogram, fade_bars) -> bool:
    if fade_bars <= 1 or len(recent_histogram) < fade_bars:
        return False
    values = np.asarray(recent_histogram[-fade_bars:], dtype="float64")
    if np.isnan(values).any() or np.any(values <= 0):
        return False
    return bool(np.all(np.diff(values) < 0))


def is_trend_intact(current_dif, current_dea, current_price, trend_ma_value) -> bool:
    if any(np.isnan(float(value)) for value in [current_dif, current_dea, current_price]):
        return False
    if current_dif <= 0 or current_dea <= 0:
        return False
    if np.isnan(float(trend_ma_value)):
        return True
    return current_price >= trend_ma_value


def get_macd_volume_exit_reason(
    previous_dif,
    previous_dea,
    current_dif,
    current_dea,
    recent_histogram,
    current_price,
    entry_price,
    highest_price,
    trend_ma_value,
    holding_bars,
    histogram_fade_bars,
    stop_loss_pct,
    take_profit_pct,
    trailing_stop_pct,
    max_holding_bars,
):
    if is_dead_cross(previous_dif, previous_dea, current_dif, current_dea):
        return "dead_cross"
    if entry_price is None:
        return None
    if stop_loss_pct and current_price <= entry_price * (1 - stop_loss_pct / 100):
        return "stop_loss"
    if (
        trailing_stop_pct
        and highest_price
        and current_price <= highest_price * (1 - trailing_stop_pct / 100)
    ):
        return "trailing_stop"

    trend_intact = is_trend_intact(current_dif, current_dea, current_price, trend_ma_value)
    if is_histogram_fading(recent_histogram, histogram_fade_bars) and not trend_intact:
        return "histogram_fade"
    if not np.isnan(float(trend_ma_value)) and current_price < trend_ma_value:
        return "trend_ma_lost"
    if not trend_intact and take_profit_pct and current_price >= entry_price * (1 + take_profit_pct / 100):
        return "take_profit"
    if not trend_intact and max_holding_bars and holding_bars >= max_holding_bars:
        return "max_holding_bars"
    return None


def prepare_macd_volume_frame(data: pd.DataFrame, config: BaseModel) -> pd.DataFrame:
    resolved = MACDVolumeConfig.model_validate(config.model_dump())
    frame = data.copy()
    close = frame["Close"].astype(float).to_numpy()
    volume = frame["Volume"].astype(float)
    dif, dea, histogram = calculate_macd(
        close,
        resolved.fast_period,
        resolved.slow_period,
        resolved.signal_period,
    )
    frame["macd_dif"] = dif
    frame["macd_dea"] = dea
    frame["macd_histogram"] = histogram
    frame["average_volume"] = volume.rolling(resolved.volume_lookback).mean()
    frame["trend_ma_value"] = frame["Close"].astype(float).rolling(resolved.trend_ma).mean()
    return frame


def evaluate_macd_volume(context: StrategyBarContext) -> StrategyDecision:
    config = MACDVolumeConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < macd_volume_min_history_bars(config):
        return StrategyDecision()
    previous = context.previous
    if previous is None:
        return StrategyDecision()
    current = context.current
    previous_dif = float(previous["macd_dif"])
    previous_dea = float(previous["macd_dea"])
    current_dif = float(current["macd_dif"])
    current_dea = float(current["macd_dea"])
    current_price = float(current["Close"])
    trend_value = float(current["trend_ma_value"])

    if context.position is not None:
        highest_price = max(
            context.position.highest_price or current_price,
            current_price,
        )
        reason = get_macd_volume_exit_reason(
            previous_dif=previous_dif,
            previous_dea=previous_dea,
            current_dif=current_dif,
            current_dea=current_dea,
            recent_histogram=context.history["macd_histogram"].iloc[-config.histogram_fade_bars :].to_numpy(),
            current_price=current_price,
            entry_price=context.position.entry_price,
            highest_price=highest_price,
            trend_ma_value=trend_value,
            holding_bars=context.position.holding_bars,
            histogram_fade_bars=config.histogram_fade_bars,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            trailing_stop_pct=config.trailing_stop_pct,
            max_holding_bars=config.max_holding_bars,
        )
        return StrategyDecision(
            exit=ExitIntent(reason) if reason else None,
            next_state={"highest_price": highest_price},
        )

    history = context.history.iloc[-config.divergence_lookback :]
    divergence = has_bullish_macd_divergence(
        history["Close"].to_numpy(),
        history["macd_dif"].to_numpy(),
        history["macd_histogram"].to_numpy(),
    )
    average_volume = float(previous["average_volume"])
    current_volume = float(current["Volume"])
    golden_entry = should_enter_macd_volume(
        previous_dif=previous_dif,
        previous_dea=previous_dea,
        current_dif=current_dif,
        current_dea=current_dea,
        current_close=current_price,
        volume=current_volume,
        average_volume=average_volume,
        volume_multiplier=config.volume_multiplier,
        divergence_detected=divergence,
        zero_axis_threshold=config.zero_axis_threshold,
    )
    continuation_entry = should_enter_continuation(
        current_dif=current_dif,
        current_dea=current_dea,
        current_close=current_price,
        trend_ma_value=trend_value,
        volume=current_volume,
        average_volume=average_volume,
        continuation_volume_multiplier=config.continuation_volume_multiplier,
        continuation_pullback_pct=config.continuation_pullback_pct,
    )
    if golden_entry or continuation_entry:
        return StrategyDecision(
            entry=EntryIntent(
                order_type="next_open",
                suggested_position_pct=config.position_pct,
                metadata={
                    "stop_loss_pct": config.stop_loss_pct,
                    "take_profit_pct": config.take_profit_pct,
                    "trailing_stop_pct": config.trailing_stop_pct,
                },
            ),
            next_state={"highest_price": current_price},
        )
    return StrategyDecision()


def macd_volume_min_history_bars(config: BaseModel) -> int:
    resolved = MACDVolumeConfig.model_validate(config.model_dump())
    return max(
        resolved.slow_period + resolved.signal_period,
        resolved.volume_lookback,
        resolved.divergence_lookback,
        resolved.trend_ma,
    ) + 2


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="macd_volume_divergence_risk_control",
    display_name="MACD放量背离风控策略",
    description="底背离或零轴附近 MACD 金叉并放量时买入，死叉、红柱衰减、跌破趋势线或触发风控时卖出。",
    config_model=MACDVolumeConfig,
    parameters=(
        StrategyParamMeta(name="fast_period", label="MACD快线周期", type="int", default=12, search_values=[8, 12, 16], min_value=2, max_value=60, step=1),
        StrategyParamMeta(name="slow_period", label="MACD慢线周期", type="int", default=26, search_values=[20, 26, 34], min_value=5, max_value=120, step=1),
        StrategyParamMeta(name="signal_period", label="DEA平滑周期", type="int", default=9, search_values=[6, 9, 12], min_value=2, max_value=40, step=1),
        StrategyParamMeta(name="volume_lookback", label="均量窗口", type="int", default=20, search_values=[10, 20, 30], min_value=3, max_value=120, step=1),
        StrategyParamMeta(name="volume_multiplier", label="放量倍数", type="float", default=2.0, search_values=[1.5, 2.0, 2.5], min_value=1, max_value=10, step=0.1),
        StrategyParamMeta(name="continuation_volume_multiplier", label="中继放量倍数", type="float", default=1.2, search_values=[1.0, 1.2, 1.5], min_value=0.8, max_value=5, step=0.1),
        StrategyParamMeta(name="continuation_pullback_pct", label="中继离趋势线", type="float", default=8, search_values=[6, 8, 12], min_value=0, max_value=30, step=0.5),
        StrategyParamMeta(name="divergence_lookback", label="背离窗口", type="int", default=60, search_values=[30, 60, 90], min_value=10, max_value=180, step=1),
        StrategyParamMeta(name="zero_axis_threshold", label="零轴距离阈值", type="float", default=0.03, search_values=[0.02, 0.03, 0.05], min_value=0.001, max_value=0.2, step=0.001),
        StrategyParamMeta(name="trend_ma", label="风险趋势线", type="int", default=60, search_values=[30, 60, 90], min_value=5, max_value=250, step=1),
        StrategyParamMeta(name="histogram_fade_bars", label="红柱衰减根数", type="int", default=3, search_values=[3, 4, 5], min_value=2, max_value=10, step=1),
        StrategyParamMeta(name="stop_loss_pct", label="止损比例", type="float", default=5, search_values=[3, 5, 8], min_value=0, max_value=30, step=0.5),
        StrategyParamMeta(name="take_profit_pct", label="止盈比例", type="float", default=12, search_values=[8, 12, 20], min_value=0, max_value=80, step=0.5),
        StrategyParamMeta(name="trailing_stop_pct", label="移动止盈回撤", type="float", default=10, search_values=[8, 10, 12], min_value=0, max_value=40, step=0.5),
        StrategyParamMeta(name="max_holding_bars", label="最大持仓周期", type="int", default=80, search_values=[40, 80, 120], min_value=1, max_value=500, step=1),
        StrategyParamMeta(name="position_pct", label="仓位比例", type="float", default=0.95, search_values=[0.5, 0.8, 0.95], min_value=0.05, max_value=1, step=0.05),
    ),
    prepare_frame=prepare_macd_volume_frame,
    evaluate=evaluate_macd_volume,
    min_history_bars=macd_volume_min_history_bars,
)
