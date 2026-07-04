import numpy as np
from backtesting import Strategy
from backtesting.test import SMA


def crossed_above(previous_fast, previous_slow, current_fast, current_slow) -> bool:
    return previous_fast <= previous_slow and current_fast > current_slow


def crossed_below(previous_fast, previous_slow, current_fast, current_slow) -> bool:
    return previous_fast >= previous_slow and current_fast < current_slow


def should_enter_ma_trend(
    previous_fast_ma,
    previous_slow_ma,
    current_fast_ma,
    current_slow_ma,
    close,
    trend_ma_value,
    momentum_return,
) -> bool:
    values = [
        previous_fast_ma,
        previous_slow_ma,
        current_fast_ma,
        current_slow_ma,
        trend_ma_value,
        momentum_return,
    ]
    if any(np.isnan(float(value)) for value in values):
        return False
    if close <= trend_ma_value:
        return False
    if momentum_return <= 0:
        return False
    return crossed_above(previous_fast_ma, previous_slow_ma, current_fast_ma, current_slow_ma)


def get_ma_exit_reason(
    previous_fast_ma,
    previous_slow_ma,
    current_fast_ma,
    current_slow_ma,
    current_price,
    entry_price,
    holding_bars,
    stop_loss_pct,
    take_profit_pct,
    max_holding_bars,
):
    if crossed_below(previous_fast_ma, previous_slow_ma, current_fast_ma, current_slow_ma):
        return "dead_cross"
    if entry_price is None:
        return None
    if stop_loss_pct and current_price <= entry_price * (1 - stop_loss_pct / 100):
        return "stop_loss"
    if take_profit_pct and current_price >= entry_price * (1 + take_profit_pct / 100):
        return "take_profit"
    if max_holding_bars and holding_bars >= max_holding_bars:
        return "max_holding_bars"
    return None


class MATrendRiskControlStrategy(Strategy):
    """MA crossover strategy with trend, momentum, and fixed risk controls."""

    strategy_name = "ma_trend_risk_control"
    display_name = "均线趋势风控策略"
    description = "短均线上穿长均线且价格位于趋势均线上方时买入，死叉或触发风控时卖出。"

    fast_ma = 10
    slow_ma = 30
    trend_ma = 60
    momentum_lookback = 5
    stop_loss_pct = 5
    take_profit_pct = 12
    max_holding_bars = 80
    position_pct = 0.95

    def init(self):
        close = self.data.Close
        self.fast = self.I(SMA, close, self.fast_ma)
        self.slow = self.I(SMA, close, self.slow_ma)
        self.trend = self.I(SMA, close, self.trend_ma)
        self.entry_price = None
        self.entry_bar = None

    def next(self):
        min_bars = max(self.slow_ma, self.trend_ma, self.momentum_lookback) + 2
        if len(self.data.Close) < min_bars:
            return

        current_price = float(self.data.Close[-1])
        previous_fast = float(self.fast[-2])
        previous_slow = float(self.slow[-2])
        current_fast = float(self.fast[-1])
        current_slow = float(self.slow[-1])
        trend_value = float(self.trend[-1])
        lookback_price = float(self.data.Close[-self.momentum_lookback])
        momentum_return = current_price / lookback_price - 1

        if self.position:
            holding_bars = len(self.data.Close) - (self.entry_bar or len(self.data.Close))
            reason = get_ma_exit_reason(
                previous_fast_ma=previous_fast,
                previous_slow_ma=previous_slow,
                current_fast_ma=current_fast,
                current_slow_ma=current_slow,
                current_price=current_price,
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
            return

        if should_enter_ma_trend(
            previous_fast_ma=previous_fast,
            previous_slow_ma=previous_slow,
            current_fast_ma=current_fast,
            current_slow_ma=current_slow,
            close=current_price,
            trend_ma_value=trend_value,
            momentum_return=momentum_return,
        ):
            self.buy(size=self.position_pct)
            self.entry_price = current_price
            self.entry_bar = len(self.data.Close)
