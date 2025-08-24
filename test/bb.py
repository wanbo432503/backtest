import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG

# 已修正的布林带计算函数
def bollinger_bands(price_series: pd.Series, n=20, k=2):
    """
    计算布林带指标。
    这个函数现在返回一个包含上、中、下轨三个Series的元组，
    以兼容 backtesting.py 的 self.I() 方法。
    """
    price_series = pd.Series(price_series)
    sma = price_series.rolling(n).mean()
    std = price_series.rolling(n).std()
    upper_band = sma + (std * k)
    lower_band = sma - (std * k)
    # 返回一个元组，而不是DataFrame
    return upper_band, sma, lower_band

class BollingerBandsStrategy(Strategy):
    """
    布林带交易策略
    - 价格跌破下轨时买入 (Price crosses under the lower band)
    - 价格突破上轨时卖出 (Price crosses over the upper band)
    """
    # 策略参数
    bb_period = 20  # 布林带周期
    bb_std = 2      # 布林带标准差倍数

    def init(self):
        # 初始化指标
        # self.I 会将 bollinger_bands 返回的元组解包并赋给 self.bb_bands
        # self.bb_bands[0] -> upper_band
        # self.bb_bands[1] -> sma
        # self.bb_bands[2] -> lower_band
        self.bb_bands = self.I(bollinger_bands, self.data.Close, self.bb_period, self.bb_std)

    def next(self):
        # 获取上轨和下轨的当前值
        upper_band = self.bb_bands[0]
        lower_band = self.bb_bands[2]

        # 买入逻辑：当价格从上方向下穿过（跌破）下轨时
        if crossover(self.data.Close, lower_band):
            # 如果当前没有持仓，则买入
            if not self.position:
                self.buy()

        # 卖出逻辑：当价格从下方向上穿过（突破）上轨时
        elif crossover(self.data.Close, upper_band):
            # 如果当前有持仓，则平仓卖出
            if self.position:
                self.position.close()


# --- 回测执行 ---

# 使用 backtesting.py 自带的谷歌股票数据作为示例
# 您可以替换成自己的数据，确保它是一个包含 'Open', 'High', 'Low', 'Close' 列的DataFrame
data = GOOG.copy()

# 初始化回测
# cash: 初始资金
# commission: 手续费率
bt = Backtest(data, BollingerBandsStrategy, cash=10000, commission=.002)

# 运行回测
stats = bt.run()

# 打印回测结果
print("--- 回测性能指标 ---")
print(stats)
print("\n--- 策略交易记录 ---")
print(stats['_trades'])


# 绘制回测结果图表
# plot=False 可以禁止自动在浏览器中打开图表
bt.plot()