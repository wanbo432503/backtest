import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.test import SMA


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


class MACDVolumeDivergenceRiskControlStrategy(Strategy):
    """MACD divergence and volume-confirmed golden-cross strategy with risk exits."""

    strategy_name = "macd_volume_divergence_risk_control"
    display_name = "MACD放量背离风控策略"
    description = "底背离或零轴附近 MACD 金叉并放量时买入，死叉、红柱衰减、跌破趋势线或触发风控时卖出。"

    fast_period = 12
    slow_period = 26
    signal_period = 9
    volume_lookback = 20
    volume_multiplier = 2.0
    continuation_volume_multiplier = 1.2
    continuation_pullback_pct = 8
    divergence_lookback = 60
    zero_axis_threshold = 0.03
    trend_ma = 60
    histogram_fade_bars = 3
    stop_loss_pct = 5
    take_profit_pct = 12
    trailing_stop_pct = 10
    max_holding_bars = 80
    position_pct = 0.95

    def init(self):
        close = self.data.Close
        volume = self.data.Volume
        self.dif = self.I(macd_dif, close, self.fast_period, self.slow_period, self.signal_period)
        self.dea = self.I(macd_dea, close, self.fast_period, self.slow_period, self.signal_period)
        self.histogram = self.I(macd_histogram, close, self.fast_period, self.slow_period, self.signal_period)
        self.average_volume = self.I(rolling_mean, volume, self.volume_lookback)
        self.trend = self.I(SMA, close, self.trend_ma)
        self.entry_price = None
        self.entry_bar = None
        self.highest_price = None

    def next(self):
        min_bars = max(
            self.slow_period + self.signal_period,
            self.volume_lookback,
            self.divergence_lookback,
            self.trend_ma,
        ) + 2
        if len(self.data.Close) < min_bars:
            return

        current_price = float(self.data.Close[-1])
        previous_dif = float(self.dif[-2])
        previous_dea = float(self.dea[-2])
        current_dif = float(self.dif[-1])
        current_dea = float(self.dea[-1])
        average_volume = float(self.average_volume[-2])
        current_volume = float(self.data.Volume[-1])
        trend_value = float(self.trend[-1])

        if self.position:
            self.highest_price = max(self.highest_price or current_price, current_price)
            holding_bars = len(self.data.Close) - (self.entry_bar or len(self.data.Close))
            reason = get_macd_volume_exit_reason(
                previous_dif=previous_dif,
                previous_dea=previous_dea,
                current_dif=current_dif,
                current_dea=current_dea,
                recent_histogram=self.histogram[-self.histogram_fade_bars :],
                current_price=current_price,
                entry_price=self.entry_price,
                highest_price=self.highest_price,
                trend_ma_value=trend_value,
                holding_bars=holding_bars,
                histogram_fade_bars=self.histogram_fade_bars,
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                trailing_stop_pct=self.trailing_stop_pct,
                max_holding_bars=self.max_holding_bars,
            )
            if reason:
                self.position.close()
                self.entry_price = None
                self.entry_bar = None
                self.highest_price = None
            return

        divergence_detected = has_bullish_macd_divergence(
            closes=self.data.Close[-self.divergence_lookback :],
            dif_values=self.dif[-self.divergence_lookback :],
            histogram_values=self.histogram[-self.divergence_lookback :],
        )
        golden_entry = should_enter_macd_volume(
            previous_dif=previous_dif,
            previous_dea=previous_dea,
            current_dif=current_dif,
            current_dea=current_dea,
            current_close=current_price,
            volume=current_volume,
            average_volume=average_volume,
            volume_multiplier=self.volume_multiplier,
            divergence_detected=divergence_detected,
            zero_axis_threshold=self.zero_axis_threshold,
        )
        continuation_entry = should_enter_continuation(
            current_dif=current_dif,
            current_dea=current_dea,
            current_close=current_price,
            trend_ma_value=trend_value,
            volume=current_volume,
            average_volume=average_volume,
            continuation_volume_multiplier=self.continuation_volume_multiplier,
            continuation_pullback_pct=self.continuation_pullback_pct,
        )
        if golden_entry or continuation_entry:
            self.buy(size=self.position_pct)
            self.entry_price = current_price
            self.entry_bar = len(self.data.Close)
            self.highest_price = current_price
