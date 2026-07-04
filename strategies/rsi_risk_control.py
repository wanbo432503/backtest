import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.test import SMA


def calculate_rsi(values, period):
    prices = pd.Series(values, dtype="float64")
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).to_numpy()


def crossed_above(previous_value, current_value, threshold) -> bool:
    return previous_value <= threshold < current_value


def should_enter_long(
    previous_rsi,
    current_rsi,
    close,
    trend_ma_value,
    cooldown_remaining,
    rsi_buy,
) -> bool:
    if cooldown_remaining > 0:
        return False
    if np.isnan(trend_ma_value) or close <= trend_ma_value:
        return False
    return crossed_above(previous_rsi, current_rsi, rsi_buy)


def get_exit_reason(
    current_price,
    entry_price,
    previous_rsi,
    current_rsi,
    close,
    trend_ma_value,
    holding_bars,
    rsi_sell,
    stop_loss_pct,
    take_profit_pct,
    max_holding_bars,
):
    if entry_price is None:
        return None
    if stop_loss_pct and current_price <= entry_price * (1 - stop_loss_pct / 100):
        return "stop_loss"
    if take_profit_pct and current_price >= entry_price * (1 + take_profit_pct / 100):
        return "take_profit"
    if max_holding_bars and holding_bars >= max_holding_bars:
        return "max_holding_bars"
    if crossed_above(previous_rsi, current_rsi, rsi_sell):
        return "rsi_sell"
    if not np.isnan(trend_ma_value) and close < trend_ma_value:
        return "trend_break"
    return None


class RSIRiskControlStrategy(Strategy):
    """RSI + trend filter strategy with fixed risk controls."""

    strategy_name = "rsi_risk_control"
    display_name = "RSI风控策略"
    description = "RSI 上穿买入阈值且处于趋势均线上方时买入，并使用止损、止盈、持仓周期和冷却期控制风险。"

    rsi_period = 14
    rsi_buy = 30
    rsi_sell = 70
    trend_ma = 60
    stop_loss_pct = 5
    take_profit_pct = 12
    max_holding_bars = 120
    position_pct = 0.95
    cooldown_bars = 3

    def init(self):
        close = self.data.Close
        self.rsi = self.I(calculate_rsi, close, self.rsi_period)
        self.trend = self.I(SMA, close, self.trend_ma)
        self.entry_price = None
        self.entry_bar = None
        self.cooldown_remaining = 0

    def next(self):
        if len(self.data.Close) < max(self.rsi_period + 2, self.trend_ma + 1):
            return

        current_price = float(self.data.Close[-1])
        previous_rsi = float(self.rsi[-2])
        current_rsi = float(self.rsi[-1])
        trend_value = float(self.trend[-1])

        if self.position:
            holding_bars = len(self.data.Close) - (self.entry_bar or len(self.data.Close))
            reason = get_exit_reason(
                current_price=current_price,
                entry_price=self.entry_price,
                previous_rsi=previous_rsi,
                current_rsi=current_rsi,
                close=current_price,
                trend_ma_value=trend_value,
                holding_bars=holding_bars,
                rsi_sell=self.rsi_sell,
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                max_holding_bars=self.max_holding_bars,
            )
            if reason:
                self.position.close()
                self.entry_price = None
                self.entry_bar = None
                self.cooldown_remaining = self.cooldown_bars
            return

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return

        if should_enter_long(
            previous_rsi=previous_rsi,
            current_rsi=current_rsi,
            close=current_price,
            trend_ma_value=trend_value,
            cooldown_remaining=self.cooldown_remaining,
            rsi_buy=self.rsi_buy,
        ):
            self.buy(size=self.position_pct)
            self.entry_price = current_price
            self.entry_bar = len(self.data.Close)
