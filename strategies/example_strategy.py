from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

class ExampleCustomStrategy(Strategy):
    """
    示例自定义策略 - 三重均线交叉策略
    短期、中期、长期均线交叉信号
    """
    
    # 策略参数
    short_period = 5
    medium_period = 20
    long_period = 60
    
    def init(self):
        close = self.data.Close
        # 计算三条均线
        self.sma_short = self.I(SMA, close, self.short_period)
        self.sma_medium = self.I(SMA, close, self.medium_period)
        self.sma_long = self.I(SMA, close, self.long_period)
    
    def next(self):
        # 三重均线金叉：短期上穿中期，且中期上穿长期
        if (crossover(self.sma_short, self.sma_medium) and 
            crossover(self.sma_medium, self.sma_long)):
            self.buy()
        
        # 三重均线死叉：短期下穿中期，且中期下穿长期
        elif (crossover(self.sma_medium, self.sma_short) and 
              crossover(self.sma_long, self.sma_medium)):
            self.sell()