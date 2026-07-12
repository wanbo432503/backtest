from __future__ import annotations

import math

import numpy as np
import pandas as pd
from backtesting import Strategy
from pydantic import BaseModel, ConfigDict, Field, model_validator

from strategy_engine import (
    EntryIntent,
    ExitIntent,
    RiskIntent,
    SimulationPosition,
    StrategyBarContext,
    StrategyDecision,
    StrategyDefinition,
    StrategyParamMeta,
)


class TrendPullbackPinBarConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_ma_period: int = Field(default=20, ge=2, le=80)
    medium_ma_period: int = Field(default=60, ge=3, le=180)
    long_ma_period: int = Field(default=120, ge=4, le=250)
    ma_distance_pct: float = Field(default=2.0, gt=0, le=20)
    support_lookback: int = Field(default=20, ge=2, le=180)
    support_tolerance_pct: float = Field(default=1.0, gt=0, le=20)
    lower_shadow_body_ratio: float = Field(default=2.5, ge=1, le=20)
    max_body_range_pct: float = Field(default=30.0, gt=0, le=100)
    min_close_location_pct: float = Field(default=65.0, gt=0, le=100)
    max_upper_shadow_range_pct: float = Field(default=20.0, gt=0, le=100)
    volume_lookback: int = Field(default=20, ge=2, le=120)
    volume_multiplier: float = Field(default=1.3, ge=1, le=10)
    atr_period: int = Field(default=14, ge=2, le=80)
    min_stop_distance_pct: float = Field(default=1.5, gt=0, le=100)
    max_stop_distance_pct: float = Field(default=6.0, gt=0, le=100)
    reward_risk_ratio: float = Field(default=2.5, ge=2, le=3)
    max_entry_gap_pct: float = Field(default=2.0, gt=0, le=20)
    risk_per_trade_pct: float = Field(default=0.5, gt=0, le=5)
    trend_exit_confirmation_days: int = Field(default=2, ge=1, le=5)
    cooldown_days: int = Field(default=20, ge=0, le=252)
    price_tick: float = Field(default=0.01, gt=0, le=1)

    @model_validator(mode="after")
    def validate_parameters(self) -> "TrendPullbackPinBarConfig":
        periods = [
            self.short_ma_period,
            self.medium_ma_period,
            self.long_ma_period,
        ]
        if periods != sorted(periods) or len(set(periods)) != 3:
            raise ValueError("moving-average periods must be strictly increasing")
        if self.min_stop_distance_pct >= self.max_stop_distance_pct:
            raise ValueError("min_stop_distance_pct must be below max_stop_distance_pct")
        return self


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def prepare_pin_bar_frame(data: pd.DataFrame, config: BaseModel) -> pd.DataFrame:
    resolved = TrendPullbackPinBarConfig.model_validate(config.model_dump())
    frame = data.copy()
    close = frame["Close"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    volume = frame["Volume"].astype(float)
    frame["ma_short_value"] = close.rolling(resolved.short_ma_period).mean()
    frame["ma_medium_value"] = close.rolling(resolved.medium_ma_period).mean()
    frame["ma_long_value"] = close.rolling(resolved.long_ma_period).mean()
    frame["support"] = low.shift(1).rolling(resolved.support_lookback).min()
    frame["average_volume"] = volume.shift(1).rolling(resolved.volume_lookback).mean()
    frame["atr"] = calculate_atr(high, low, close, resolved.atr_period)
    return frame


def is_trend_pullback_pin_bar(
    current: pd.Series,
    config: TrendPullbackPinBarConfig,
) -> bool:
    required = [
        current["Open"],
        current["High"],
        current["Low"],
        current["Close"],
        current["Volume"],
        current["ma_short_value"],
        current["ma_medium_value"],
        current["ma_long_value"],
        current["support"],
        current["average_volume"],
        current["atr"],
    ]
    if not all(math.isfinite(float(value)) for value in required):
        return False
    open_price = float(current["Open"])
    high = float(current["High"])
    low = float(current["Low"])
    close = float(current["Close"])
    short_ma = float(current["ma_short_value"])
    medium_ma = float(current["ma_medium_value"])
    long_ma = float(current["ma_long_value"])
    support = float(current["support"])
    candle_range = high - low
    if candle_range <= 0 or short_ma <= 0 or support <= 0:
        return False

    body = abs(close - open_price)
    lower_shadow = min(open_price, close) - low
    upper_shadow = high - max(open_price, close)
    trend_ok = short_ma > medium_ma > long_ma and close > medium_ma
    near_ma = abs(close / short_ma - 1) * 100 <= config.ma_distance_pct
    near_support = abs(low / support - 1) * 100 <= config.support_tolerance_pct
    pin_bar_ok = (
        lower_shadow >= body * config.lower_shadow_body_ratio
        and body / candle_range * 100 <= config.max_body_range_pct
        and (close - low) / candle_range * 100 >= config.min_close_location_pct
        and upper_shadow / candle_range * 100 <= config.max_upper_shadow_range_pct
    )
    volume_ok = (
        float(current["average_volume"]) > 0
        and float(current["Volume"])
        >= float(current["average_volume"]) * config.volume_multiplier
    )
    return trend_ok and (near_ma or near_support) and pin_bar_ok and volume_ok


def pin_bar_signal_strength(current: pd.Series) -> float:
    candle_range = float(current["High"]) - float(current["Low"])
    volume_ratio = float(current["Volume"]) / float(current["average_volume"])
    close_location = (float(current["Close"]) - float(current["Low"])) / candle_range
    trend_spread = (
        float(current["ma_short_value"]) / float(current["ma_medium_value"]) - 1
        + float(current["ma_medium_value"]) / float(current["ma_long_value"]) - 1
    )
    return volume_ratio * 0.5 + close_location * 0.3 + trend_spread * 20


def evaluate_pin_bar(context: StrategyBarContext) -> StrategyDecision:
    config = TrendPullbackPinBarConfig.model_validate(context.config.model_dump())
    if context.bar_index + 1 < pin_bar_min_history_bars(config):
        return StrategyDecision()
    current = context.current
    state = dict(context.state)
    if context.position is not None:
        weak_bars = int(state.get("trend_weak_bars", 0))
        if float(current["Close"]) < float(current["ma_short_value"]):
            weak_bars += 1
        else:
            weak_bars = 0
        next_state = {"trend_weak_bars": weak_bars}
        if weak_bars >= config.trend_exit_confirmation_days:
            return StrategyDecision(
                exit=ExitIntent("trend_weak"),
                next_state=next_state,
            )
        return StrategyDecision(next_state=next_state)

    if not is_trend_pullback_pin_bar(current, config):
        return StrategyDecision(next_state={"trend_weak_bars": 0})
    trigger_price = float(current["High"])
    structural_stop = round(float(current["Low"]) - config.price_tick, 6)
    atr_stop = trigger_price - float(current["atr"])
    stop_price = round(min(structural_stop, atr_stop), 6)
    risk_per_share = trigger_price - stop_price
    if trigger_price <= 0 or risk_per_share <= 0:
        return StrategyDecision()
    stop_distance_pct = risk_per_share / trigger_price * 100
    if not config.min_stop_distance_pct <= stop_distance_pct <= config.max_stop_distance_pct:
        return StrategyDecision()
    target_price = round(
        trigger_price + risk_per_share * config.reward_risk_ratio,
        6,
    )
    return StrategyDecision(
        entry=EntryIntent(
            order_type="stop_next_bar",
            strength=pin_bar_signal_strength(current),
            trigger_price=trigger_price,
            expires_after_bars=1,
            suggested_position_pct=1.0,
            risk=RiskIntent(
                stop_price=stop_price,
                target_price=target_price,
                risk_per_share=risk_per_share,
                risk_budget_pct=config.risk_per_trade_pct / 100,
            ),
            metadata={"max_entry_gap_pct": config.max_entry_gap_pct},
        ),
        next_state={"trend_weak_bars": 0},
    )


def pin_bar_min_history_bars(config: BaseModel) -> int:
    resolved = TrendPullbackPinBarConfig.model_validate(config.model_dump())
    return max(
        resolved.long_ma_period,
        resolved.support_lookback + 1,
        resolved.volume_lookback + 1,
        resolved.atr_period + 1,
    )


def _param(
    name: str,
    label: str,
    type_: str,
    default,
    *,
    minimum=None,
    maximum=None,
    step=None,
    search_values=None,
) -> StrategyParamMeta:
    return StrategyParamMeta(
        name=name,
        label=label,
        type=type_,
        default=default,
        search_values=search_values or [default],
        min_value=minimum,
        max_value=maximum,
        step=step,
    )


STRATEGY_DEFINITION = StrategyDefinition(
    strategy_id="trend_pullback_pin_bar",
    display_name="趋势回调 Pin Bar 策略",
    description="多头均线趋势中等待价格回调至短均线或支撑位，放量看涨 Pin Bar 后于下一交易日突破形态高点买入。",
    config_model=TrendPullbackPinBarConfig,
    parameters=(
        _param("short_ma_period", "短期均线", "int", 20, minimum=2, maximum=80, step=1, search_values=[10, 20, 30]),
        _param("medium_ma_period", "中期均线", "int", 60, minimum=3, maximum=180, step=1, search_values=[40, 60, 90]),
        _param("long_ma_period", "长期均线", "int", 120, minimum=4, maximum=250, step=1, search_values=[90, 120, 200]),
        _param("ma_distance_pct", "距短均线上限", "float", 2.0, minimum=0.1, maximum=20, step=0.1),
        _param("support_lookback", "支撑回看天数", "int", 20, minimum=2, maximum=180, step=1),
        _param("support_tolerance_pct", "支撑距离", "float", 1.0, minimum=0.1, maximum=20, step=0.1),
        _param("lower_shadow_body_ratio", "下影/实体倍数", "float", 2.5, minimum=1, maximum=20, step=0.1),
        _param("max_body_range_pct", "实体占比上限", "float", 30.0, minimum=0.1, maximum=100, step=1),
        _param("min_close_location_pct", "收盘位置下限", "float", 65.0, minimum=0.1, maximum=100, step=1),
        _param("max_upper_shadow_range_pct", "上影占比上限", "float", 20.0, minimum=0.1, maximum=100, step=1),
        _param("volume_lookback", "均量窗口", "int", 20, minimum=2, maximum=120, step=1),
        _param("volume_multiplier", "量能倍数", "float", 1.3, minimum=1, maximum=10, step=0.1),
        _param("atr_period", "ATR周期", "int", 14, minimum=2, maximum=80, step=1),
        _param("min_stop_distance_pct", "最小止损距离", "float", 1.5, minimum=0.1, maximum=100, step=0.1),
        _param("max_stop_distance_pct", "最大止损距离", "float", 6.0, minimum=0.1, maximum=100, step=0.1),
        _param("reward_risk_ratio", "止盈R倍数", "float", 2.5, minimum=2, maximum=3, step=0.5),
        _param("max_entry_gap_pct", "最大高开", "float", 2.0, minimum=0.1, maximum=20, step=0.1),
        _param("risk_per_trade_pct", "单笔账户风险", "float", 0.5, minimum=0.1, maximum=5, step=0.1),
        _param("trend_exit_confirmation_days", "趋势退出确认", "int", 2, minimum=1, maximum=5, step=1),
        _param("cooldown_days", "同票冷却天数", "int", 20, minimum=0, maximum=252, step=1),
        _param("price_tick", "最小价格变动", "float", 0.01, minimum=0.01, maximum=1, step=0.01),
    ),
    prepare_frame=prepare_pin_bar_frame,
    evaluate=evaluate_pin_bar,
    min_history_bars=pin_bar_min_history_bars,
)


class TrendPullbackPinBarStrategy(Strategy):
    """Temporary Backtesting.py wrapper around the unified Pin Bar definition."""

    strategy_name = STRATEGY_DEFINITION.strategy_id
    display_name = STRATEGY_DEFINITION.display_name
    description = STRATEGY_DEFINITION.description

    for _field_name, _field in TrendPullbackPinBarConfig.model_fields.items():
        locals()[_field_name] = _field.default

    def init(self):
        self._strategy_state = {}

    def next(self):
        raw = pd.DataFrame(
            {
                "Open": np.asarray(self.data.Open, dtype=float),
                "High": np.asarray(self.data.High, dtype=float),
                "Low": np.asarray(self.data.Low, dtype=float),
                "Close": np.asarray(self.data.Close, dtype=float),
                "Volume": np.asarray(self.data.Volume, dtype=float),
            }
        )
        config = TrendPullbackPinBarConfig(
            **{
                name: getattr(self, name)
                for name in TrendPullbackPinBarConfig.model_fields
            }
        )
        frame = prepare_pin_bar_frame(raw, config)
        position = None
        if self.position:
            trade = self.trades[-1]
            position = SimulationPosition(
                symbol="single",
                shares=int(abs(self.position.size)),
                entry_date=str(trade.entry_time),
                entry_price=float(trade.entry_price),
                holding_bars=max(0, len(frame) - int(trade.entry_bar) - 1),
            )
        decision = evaluate_pin_bar(
            StrategyBarContext(
                symbol="single",
                frame=frame,
                bar_index=len(frame) - 1,
                config=config,
                position=position,
                state=self._strategy_state,
            )
        )
        self._strategy_state = dict(decision.next_state or self._strategy_state)
        if decision.exit and self.position:
            self.position.close()
        elif decision.entry and not self.position:
            risk = decision.entry.risk
            self.buy(
                size=min(0.95, decision.entry.suggested_position_pct),
                stop=decision.entry.trigger_price,
                sl=risk.stop_price if risk else None,
                tp=risk.target_price if risk else None,
            )
