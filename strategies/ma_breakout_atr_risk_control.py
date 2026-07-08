import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.test import SMA


def rolling_max(values, window):
    return pd.Series(values, dtype="float64").rolling(window).max().to_numpy()


def rolling_mean(values, window):
    return pd.Series(values, dtype="float64").rolling(window).mean().to_numpy()


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


class MABreakoutATRRiskControlStrategy(Strategy):
    """MA trend breakout strategy with ATR trailing stop and volatility sizing."""

    strategy_name = "ma_breakout_atr_risk_control"
    display_name = "均线突破ATR风控策略"
    description = "MA120 多头过滤、MA20/MA60 趋势确认、40日高点放量突破买入，并使用 ATR 移动止损和波动率仓位控制。"

    short_ma = 20
    medium_ma = 60
    long_ma = 120
    breakout_lookback = 40
    volume_lookback = 20
    volume_multiplier = 1.5
    atr_period = 14
    atr_stop_multiplier = 2.5
    max_holding_bars = 80
    target_atr_risk_pct = 0.02
    min_position_pct = 0.2
    max_position_pct = 0.95

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        self.ma20 = self.I(SMA, close, self.short_ma)
        self.ma60 = self.I(SMA, close, self.medium_ma)
        self.ma120 = self.I(SMA, close, self.long_ma)
        self.highest_high = self.I(rolling_max, high, self.breakout_lookback)
        self.average_volume = self.I(rolling_mean, volume, self.volume_lookback)
        self.atr = self.I(calculate_atr, high, low, close, self.atr_period)
        self.entry_bar = None
        self.highest_close = None

    def next(self):
        min_bars = max(
            self.long_ma,
            self.breakout_lookback,
            self.volume_lookback,
            self.atr_period,
        ) + 2
        if len(self.data.Close) < min_bars:
            return

        current_close = float(self.data.Close[-1])
        current_ma20 = float(self.ma20[-1])
        current_ma60 = float(self.ma60[-1])
        current_ma120 = float(self.ma120[-1])
        previous_highest_high = float(self.highest_high[-2])
        current_volume = float(self.data.Volume[-1])
        average_volume = float(self.average_volume[-2])
        current_atr = float(self.atr[-1])

        if self.position:
            self.highest_close = max(self.highest_close or current_close, current_close)
            holding_bars = len(self.data.Close) - (self.entry_bar or len(self.data.Close))
            reason = get_ma_breakout_atr_exit_reason(
                close=current_close,
                ma20=current_ma20,
                ma60=current_ma60,
                highest_close=self.highest_close,
                atr=current_atr,
                atr_stop_multiplier=self.atr_stop_multiplier,
                holding_bars=holding_bars,
                max_holding_bars=self.max_holding_bars,
            )
            if reason:
                self.position.close()
                self.entry_bar = None
                self.highest_close = None
            return

        if should_enter_ma_breakout(
            close=current_close,
            ma20=current_ma20,
            ma60=current_ma60,
            ma120=current_ma120,
            previous_highest_high=previous_highest_high,
            volume=current_volume,
            average_volume=average_volume,
            volume_multiplier=self.volume_multiplier,
        ):
            position_pct = calculate_atr_position_pct(
                close=current_close,
                atr=current_atr,
                target_atr_risk_pct=self.target_atr_risk_pct,
                min_position_pct=self.min_position_pct,
                max_position_pct=self.max_position_pct,
            )
            self.buy(size=position_pct)
            self.entry_bar = len(self.data.Close)
            self.highest_close = current_close
