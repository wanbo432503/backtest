from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from strategy_engine import (
    EntryIntent,
    ExitIntent,
    RiskIntent,
    StrategyBarContext,
    StrategyDecision,
    StrategyDefinition,
    StrategyParamMeta,
)


class VolumeDivergenceRSILongConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ma_period: int = Field(default=20, ge=2, le=250)
    volume_lookback: int = Field(default=20, ge=2, le=120)
    volume_multiplier: float = Field(default=1.2, ge=1, le=10)
    macd_fast_period: int = Field(default=12, ge=2, le=60)
    macd_slow_period: int = Field(default=26, ge=3, le=120)
    divergence_lookback: int = Field(default=30, ge=6, le=180)
    divergence_valid_bars: int = Field(default=10, ge=1, le=60)
    rsi_period: int = Field(default=14, ge=2, le=60)
    rsi_oversold: float = Field(default=30, ge=1, le=50)
    stop_loss_pct: float = Field(default=3, gt=0, le=30)
    profit_activation_pct: float = Field(default=5, gt=0, le=50)
    locked_profit_pct: float = Field(default=2, ge=0, le=30)
    trailing_stop_pct: float = Field(default=2, gt=0, le=30)
    trend_exit_confirmation_days: int = Field(default=2, ge=1, le=10)
    time_exit_bars: int = Field(default=20, ge=1, le=500)
    time_exit_min_profit_pct: float = Field(default=3, ge=0, le=30)
    position_pct: float = Field(default=0.10, ge=0.01, le=1)

    @model_validator(mode="after")
    def validate_relationships(self) -> "VolumeDivergenceRSILongConfig":
        if self.macd_fast_period >= self.macd_slow_period:
            raise ValueError("macd_fast_period must be below macd_slow_period")
        if self.divergence_valid_bars > self.divergence_lookback:
            raise ValueError(
                "divergence_valid_bars must not exceed divergence_lookback"
            )
        if self.locked_profit_pct >= self.profit_activation_pct:
            raise ValueError(
                "locked_profit_pct must be below profit_activation_pct"
            )
        return self


def calculate_rsi(values, period: int) -> np.ndarray:
    prices = pd.Series(values, dtype="float64")
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0)
    rsi = rsi.mask((average_gain == 0) & (average_loss == 0), 50)
    return rsi.fillna(50).to_numpy()


def calculate_macd_dif(values, fast_period: int, slow_period: int) -> np.ndarray:
    prices = pd.Series(values, dtype="float64")
    fast = prices.ewm(
        span=fast_period,
        adjust=False,
        min_periods=fast_period,
    ).mean()
    slow = prices.ewm(
        span=slow_period,
        adjust=False,
        min_periods=slow_period,
    ).mean()
    return (fast - slow).to_numpy()


def has_macd_bottom_divergence(
    close: pd.Series,
    dif: pd.Series,
    *,
    lookback: int,
) -> bool:
    if len(close) < lookback or len(dif) < lookback:
        return False
    price_window = close.iloc[-lookback:].astype(float)
    dif_window = dif.reindex(price_window.index).astype(float)
    if price_window.isna().any() or dif_window.isna().any():
        return False
    split = lookback // 2
    first_prices = price_window.iloc[:split]
    second_prices = price_window.iloc[split:]
    if first_prices.empty or second_prices.empty:
        return False
    first_low_index = first_prices.idxmin()
    second_low_index = second_prices.idxmin()
    return (
        second_low_index == price_window.index[-1]
        and float(price_window.loc[second_low_index])
        < float(price_window.loc[first_low_index])
        and float(dif_window.loc[second_low_index])
        > float(dif_window.loc[first_low_index])
    )


def calculate_bottom_divergence_events(
    close: pd.Series,
    dif: pd.Series,
    *,
    lookback: int,
) -> pd.Series:
    events = pd.Series(False, index=close.index)
    for row_index in range(lookback - 1, len(close)):
        events.iloc[row_index] = has_macd_bottom_divergence(
            close.iloc[: row_index + 1],
            dif.iloc[: row_index + 1],
            lookback=lookback,
        )
    return events


def prepare_volume_divergence_rsi_long_frame(
    data: pd.DataFrame,
    config: BaseModel,
) -> pd.DataFrame:
    resolved = VolumeDivergenceRSILongConfig.model_validate(config.model_dump())
    frame = data.copy()
    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float)
    frame["ma_value"] = close.rolling(resolved.ma_period).mean()
    frame["average_volume"] = (
        volume.shift(1).rolling(resolved.volume_lookback).mean()
    )
    frame["volume_confirmed"] = (
        volume >= frame["average_volume"] * resolved.volume_multiplier
    )
    frame["macd_dif"] = calculate_macd_dif(
        close.to_numpy(),
        resolved.macd_fast_period,
        resolved.macd_slow_period,
    )
    divergence = calculate_bottom_divergence_events(
        frame["Close"],
        frame["macd_dif"],
        lookback=resolved.divergence_lookback,
    )
    frame["bottom_divergence"] = divergence
    frame["bottom_divergence_recent"] = (
        divergence.astype(int)
        .rolling(resolved.divergence_valid_bars, min_periods=1)
        .max()
        .astype(bool)
    )
    frame["rsi"] = calculate_rsi(close.to_numpy(), resolved.rsi_period)
    return frame


def evaluate_volume_divergence_rsi_long(
    context: StrategyBarContext,
) -> StrategyDecision:
    config = VolumeDivergenceRSILongConfig.model_validate(
        context.config.model_dump()
    )
    if context.bar_index + 1 < volume_divergence_rsi_long_min_history(config):
        return StrategyDecision()
    current = context.current
    previous = context.previous
    if previous is None:
        return StrategyDecision()

    state = dict(context.state)
    if context.position is not None:
        position = context.position
        peak_price = max(
            float(state.get("peak_price", position.entry_price)),
            float(current["High"]),
        )
        weak_bars = int(state.get("trend_weak_bars", 0))
        if float(current["Close"]) < float(current["ma_value"]):
            weak_bars += 1
        else:
            weak_bars = 0
        next_state = {
            "peak_price": peak_price,
            "trend_weak_bars": weak_bars,
        }
        if weak_bars >= config.trend_exit_confirmation_days:
            return StrategyDecision(
                exit=ExitIntent("trend_break"),
                next_state=next_state,
            )
        maximum_profit_pct = (
            peak_price / position.entry_price - 1
        ) * 100
        if (
            position.holding_bars >= config.time_exit_bars
            and maximum_profit_pct < config.time_exit_min_profit_pct
        ):
            return StrategyDecision(
                exit=ExitIntent("unproductive_time_exit"),
                next_state=next_state,
            )
        if maximum_profit_pct >= config.profit_activation_pct:
            existing_risk = position.risk
            existing_stop = (
                float(existing_risk.stop_price)
                if existing_risk is not None
                and existing_risk.stop_price is not None
                else position.entry_price * (1 - config.stop_loss_pct / 100)
            )
            locked_stop = position.entry_price * (
                1 + config.locked_profit_pct / 100
            )
            trailing_stop = peak_price * (1 - config.trailing_stop_pct / 100)
            stop_price = round(
                max(existing_stop, locked_stop, trailing_stop),
                6,
            )
            return StrategyDecision(
                risk_update=RiskIntent(
                    stop_price=stop_price,
                    target_price=(existing_risk.target_price if existing_risk else None),
                    risk_per_share=(existing_risk.risk_per_share if existing_risk else None),
                    risk_budget_pct=(existing_risk.risk_budget_pct if existing_risk else None),
                ),
                next_state=next_state,
            )
        return StrategyDecision(next_state=next_state)

    crossed_ma = (
        float(previous["Close"]) <= float(previous["ma_value"])
        and float(current["Close"]) > float(current["ma_value"])
    )
    crossed_rsi = (
        float(previous["rsi"]) <= config.rsi_oversold
        < float(current["rsi"])
    )
    if (
        crossed_ma
        and bool(current["volume_confirmed"])
        and bool(current["bottom_divergence_recent"])
        and crossed_rsi
    ):
        average_volume = float(current["average_volume"])
        volume_strength = (
            float(current["Volume"]) / average_volume
            if average_volume > 0
            else 0.0
        )
        return StrategyDecision(
            entry=EntryIntent(
                order_type="next_open",
                strength=volume_strength + float(current["rsi"]) / 100,
                suggested_position_pct=config.position_pct,
                metadata={"stop_loss_pct": config.stop_loss_pct},
            ),
            next_state={"peak_price": 0.0, "trend_weak_bars": 0},
        )
    return StrategyDecision(
        next_state={"peak_price": 0.0, "trend_weak_bars": 0}
    )


def volume_divergence_rsi_long_min_history(config: BaseModel) -> int:
    resolved = VolumeDivergenceRSILongConfig.model_validate(config.model_dump())
    return max(
        resolved.ma_period + 1,
        resolved.volume_lookback + 1,
        resolved.macd_slow_period + resolved.divergence_lookback - 1,
        resolved.rsi_period + 2,
    )


def _param(name, label, type_, default, values, minimum, maximum, step):
    return StrategyParamMeta(
        name=name,
        label=label,
        type=type_,
        default=default,
        search_values=values,
        min_value=minimum,
        max_value=maximum,
        step=step,
    )


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="volume_divergence_rsi_long",
    display_name="放量突破底背离 RSI 做多策略",
    description="放量突破 MA20、近期 MACD 底背离且 RSI 从超卖区向上突破时次日买入，并使用初始止损、锁盈移动保护、趋势和时间退出。",
    config_model=VolumeDivergenceRSILongConfig,
    parameters=(
        _param("ma_period", "突破均线周期", "int", 20, [10, 20, 30], 2, 250, 1),
        _param("volume_lookback", "均量窗口", "int", 20, [10, 20, 30], 2, 120, 1),
        _param("volume_multiplier", "放量倍数", "float", 1.2, [1.1, 1.2, 1.5], 1, 10, 0.1),
        _param("macd_fast_period", "MACD快线", "int", 12, [8, 12, 16], 2, 60, 1),
        _param("macd_slow_period", "MACD慢线", "int", 26, [20, 26, 35], 3, 120, 1),
        _param("divergence_lookback", "底背离窗口", "int", 30, [20, 30, 40], 6, 180, 1),
        _param("divergence_valid_bars", "底背离有效期", "int", 10, [5, 10, 15], 1, 60, 1),
        _param("rsi_period", "RSI周期", "int", 14, [6, 14, 21], 2, 60, 1),
        _param("rsi_oversold", "RSI超卖阈值", "float", 30, [25, 30, 35], 1, 50, 1),
        _param("stop_loss_pct", "初始止损", "float", 3, [2, 3, 5], 0.1, 30, 0.1),
        _param("profit_activation_pct", "移动保护激活盈利", "float", 5, [4, 5, 8], 0.1, 50, 0.1),
        _param("locked_profit_pct", "激活后锁定盈利", "float", 2, [1, 2, 3], 0, 30, 0.5),
        _param("trailing_stop_pct", "最高价回撤保护", "float", 2, [1.5, 2, 3], 0.1, 30, 0.1),
        _param("trend_exit_confirmation_days", "跌破均线确认天数", "int", 2, [1, 2, 3], 1, 10, 1),
        _param("time_exit_bars", "低效持仓退出周期", "int", 20, [10, 20, 30], 1, 500, 1),
        _param("time_exit_min_profit_pct", "有效盈利门槛", "float", 3, [2, 3, 5], 0, 30, 0.5),
        _param("position_pct", "建议仓位比例", "float", 0.10, [0.05, 0.10, 0.15], 0.01, 1, 0.01),
    ),
    prepare_frame=prepare_volume_divergence_rsi_long_frame,
    evaluate=evaluate_volume_divergence_rsi_long,
    min_history_bars=volume_divergence_rsi_long_min_history,
)
