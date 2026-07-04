import numpy as np
import pandas as pd
from backtesting import Strategy


def rolling_max(values, window):
    return pd.Series(values, dtype="float64").rolling(window).max().to_numpy()


def rolling_mean(values, window):
    return pd.Series(values, dtype="float64").rolling(window).mean().to_numpy()


def is_limit_up(close, previous_close, threshold=0.098) -> bool:
    if previous_close <= 0:
        return False
    return close / previous_close - 1 >= threshold


def should_enter_breakout(
    close,
    previous_highest_close,
    volume,
    average_volume,
    volume_multiplier,
    previous_close,
    limit_up_down_filter,
) -> bool:
    values = [close, previous_highest_close, volume, average_volume, previous_close]
    if any(np.isnan(float(value)) for value in values):
        return False
    if close <= previous_highest_close:
        return False
    if average_volume <= 0 or volume < average_volume * volume_multiplier:
        return False
    if limit_up_down_filter and is_limit_up(close, previous_close):
        return False
    return True


def get_breakout_exit_reason(
    current_price,
    breakout_line,
    entry_price,
    holding_bars,
    stop_loss_pct,
    take_profit_pct,
    max_holding_bars,
):
    if breakout_line is not None and current_price < breakout_line:
        return "breakout_line_lost"
    if entry_price is None:
        return None
    if stop_loss_pct and current_price <= entry_price * (1 - stop_loss_pct / 100):
        return "stop_loss"
    if take_profit_pct and current_price >= entry_price * (1 + take_profit_pct / 100):
        return "take_profit"
    if max_holding_bars and holding_bars >= max_holding_bars:
        return "max_holding_bars"
    return None


class VolumeBreakoutRiskControlStrategy(Strategy):
    """Volume-confirmed breakout strategy with fixed risk controls."""

    strategy_name = "volume_breakout_risk_control"
    display_name = "放量突破风控策略"
    description = "价格突破近期高点且成交量显著放大时买入，跌破突破线或触发风控时卖出。"

    breakout_lookback = 20
    volume_lookback = 10
    volume_multiplier = 2.0
    stop_loss_pct = 5
    take_profit_pct = 12
    max_holding_bars = 80
    position_pct = 0.95
    limit_up_down_filter = True

    def init(self):
        close = self.data.Close
        volume = self.data.Volume
        self.highest_close = self.I(rolling_max, close, self.breakout_lookback)
        self.average_volume = self.I(rolling_mean, volume, self.volume_lookback)
        self.entry_price = None
        self.entry_bar = None
        self.breakout_line = None

    def next(self):
        min_bars = max(self.breakout_lookback, self.volume_lookback) + 2
        if len(self.data.Close) < min_bars:
            return

        current_price = float(self.data.Close[-1])
        previous_close = float(self.data.Close[-2])
        previous_highest_close = float(self.highest_close[-2])
        current_volume = float(self.data.Volume[-1])
        average_volume = float(self.average_volume[-2])

        if self.position:
            holding_bars = len(self.data.Close) - (self.entry_bar or len(self.data.Close))
            reason = get_breakout_exit_reason(
                current_price=current_price,
                breakout_line=self.breakout_line,
                entry_price=self.entry_price,
                holding_bars=holding_bars,
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                max_holding_bars=self.max_holding_bars,
            )
            if reason:
                self.position.close()
                self.entry_price = None
                self.entry_bar = None
                self.breakout_line = None
            return

        if should_enter_breakout(
            close=current_price,
            previous_highest_close=previous_highest_close,
            volume=current_volume,
            average_volume=average_volume,
            volume_multiplier=self.volume_multiplier,
            previous_close=previous_close,
            limit_up_down_filter=self.limit_up_down_filter,
        ):
            self.buy(size=self.position_pct)
            self.entry_price = current_price
            self.entry_bar = len(self.data.Close)
            self.breakout_line = previous_highest_close
