from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd


def RSI(array, n):
    """Relative strength index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)

class SmaRsiStoploss(Strategy):
    """
    RSI & MA 交易策略（带止损）
    - RSI > 50 且价格 > MA 时买入
    - RSI < 50 且价格 < MA 时卖出
    - 设置固定百分比的止损
    """
    
    # 策略参数
    rsi_period = 14
    sma_period = 50
    rsi_buy_threshold = 50
    rsi_sell_threshold = 50
    stop_loss_pct = 0.05  # 5% 止损

    def init(self):
        # 计算指标
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        self.sma = self.I(SMA, self.data.Close, self.sma_period)
    
    def next(self):
        price = self.data.Close[-1]

        # 买入信号：无持仓，且满足RSI和MA条件
        if not self.position and self.rsi[-1] > self.rsi_buy_threshold and price > self.sma[-1]:
            # 设置止损价格并买入
            sl_price = price * (1 - self.stop_loss_pct)
            self.buy(sl=sl_price)

        # 卖出信号：有持仓（为了平掉多仓），且满足RSI和MA条件
        # 或者可以直接开空仓，这里根据常见逻辑，先平多仓
        elif self.position.is_long and (self.rsi[-1] < self.rsi_sell_threshold or price < self.sma[-1]):
            self.position.close()

        # 开空仓信号：无持仓，且满足RSI和MA条件
        elif not self.position and self.rsi[-1] < self.rsi_sell_threshold and price < self.sma[-1]:
            # 设置止损价格并卖出
            sl_price = price * (1 + self.stop_loss_pct)
            self.sell(sl=sl_price)
        
        # 平空仓信号：有空仓，且满足RSI和MA条件
        elif self.position.is_short and (self.rsi[-1] > self.rsi_buy_threshold or price > self.sma[-1]):
            self.position.close()