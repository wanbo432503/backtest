import numpy as np
import pandas as pd
from backtesting import Strategy

from strategies.macd_volume_divergence_risk_control import macd_dea, macd_dif


def bollinger_middle(values, period):
    prices = pd.Series(values, dtype="float64")
    return prices.rolling(period).mean().to_numpy()


def bollinger_upper(values, period, stddev):
    prices = pd.Series(values, dtype="float64")
    rolling = prices.rolling(period)
    return (rolling.mean() + rolling.std(ddof=0) * stddev).to_numpy()


def should_enter_boll_macd_breakout(
    previous_close,
    previous_middle,
    previous_upper,
    previous_dif,
    previous_dea,
    current_close,
    current_middle,
    current_upper,
    current_dif,
    current_dea,
) -> bool:
    values = [
        previous_close,
        previous_middle,
        previous_upper,
        previous_dif,
        previous_dea,
        current_close,
        current_middle,
        current_upper,
        current_dif,
        current_dea,
    ]
    if any(np.isnan(float(value)) for value in values):
        return False

    middle_is_rising = current_middle > previous_middle
    crossed_upper_band = previous_close <= previous_upper and current_close > current_upper
    macd_golden_cross = previous_dif <= previous_dea and current_dif > current_dea
    return middle_is_rising and crossed_upper_band and macd_golden_cross


def validate_boll_macd_risk_percentages(stop_loss_pct, take_profit_pct):
    stop_loss_pct = float(stop_loss_pct)
    take_profit_pct = float(take_profit_pct)
    risk_values = [stop_loss_pct, take_profit_pct]
    valid_values = all(
        np.isfinite(value)
        and 0.1 <= value <= 10
        and np.isclose(value * 10, round(value * 10), rtol=0, atol=1e-9)
        for value in risk_values
    )
    if not valid_values:
        raise ValueError(
            "stop_loss_pct and take_profit_pct must be between 0.1 and 10.0 "
            "in 0.1 increments"
        )
    return stop_loss_pct, take_profit_pct


def get_boll_macd_risk_prices(entry_price, stop_loss_pct, take_profit_pct):
    stop_loss_pct, take_profit_pct = validate_boll_macd_risk_percentages(
        stop_loss_pct,
        take_profit_pct,
    )
    stop_price = entry_price * (1 - stop_loss_pct / 100)
    take_price = entry_price * (1 + take_profit_pct / 100)
    return stop_price, take_price


class BollMACDBreakoutStrategy(Strategy):
    """BOLL upper-band breakout confirmed by a rising middle band and MACD golden cross."""

    strategy_name = "boll_macd_breakout"
    display_name = "BOLL+MACD上轨突破策略"
    description = "布林中轨向上、收盘价上穿上轨且MACD同步金叉时买入，按可优化比例止盈止损。"

    boll_period = 20
    boll_stddev = 2.0
    fast_period = 12
    slow_period = 26
    signal_period = 9
    stop_loss_pct = 1.0
    take_profit_pct = 1.0
    position_pct = 0.95

    def init(self):
        validate_boll_macd_risk_percentages(self.stop_loss_pct, self.take_profit_pct)
        close = self.data.Close
        self.middle = self.I(bollinger_middle, close, self.boll_period)
        self.upper = self.I(bollinger_upper, close, self.boll_period, self.boll_stddev)
        self.dif = self.I(macd_dif, close, self.fast_period, self.slow_period, self.signal_period)
        self.dea = self.I(macd_dea, close, self.fast_period, self.slow_period, self.signal_period)

    def next(self):
        min_bars = max(self.boll_period, self.slow_period + self.signal_period) + 2
        if len(self.data.Close) < min_bars:
            return

        if self.position:
            trade = self.trades[-1]
            if trade.sl is None or trade.tp is None:
                stop_price, take_price = get_boll_macd_risk_prices(
                    entry_price=float(trade.entry_price),
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                )
                # On daily bars, post-entry attachment makes exits eligible from T+1.
                trade.sl = stop_price
                trade.tp = take_price
            return

        if should_enter_boll_macd_breakout(
            previous_close=float(self.data.Close[-2]),
            previous_middle=float(self.middle[-2]),
            previous_upper=float(self.upper[-2]),
            previous_dif=float(self.dif[-2]),
            previous_dea=float(self.dea[-2]),
            current_close=float(self.data.Close[-1]),
            current_middle=float(self.middle[-1]),
            current_upper=float(self.upper[-1]),
            current_dif=float(self.dif[-1]),
            current_dea=float(self.dea[-1]),
        ):
            self.buy(size=self.position_pct)
