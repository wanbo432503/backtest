from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import numpy as np

class HighWinRateSMAStrategy(Strategy):
    """
    高胜率均线策略 - 快速追踪趋势变化
    
    特点：
    1. 使用短周期均线(5, 15)快速响应价格变化
    2. 更频繁的交易机会，提高资金利用效率
    3. 较小的止损幅度，快速止损风险
    4. 适合强势行情
    
    参数：快速(5)、中期(15)、长期(40)均线
    """
    
    # 更激进的均线周期
    fast_ma = 5       # 超快速均线
    medium_ma = 15    # 中期均线
    slow_ma = 40      # 长期均线
    
    # 风险管理 - 更激进
    stop_loss_pct = 2.0      # 2% 快速止损
    take_profit_pct = 4.0    # 4% 快速止盈
    
    # 仓位管理
    position_size = 0.95
    
    def init(self):
        """初始化策略指标"""
        close = self.data.Close
        
        # 计算三条均线
        self.sma_fast = self.I(SMA, close, self.fast_ma)
        self.sma_medium = self.I(SMA, close, self.medium_ma)
        self.sma_slow = self.I(SMA, close, self.slow_ma)
        
        self.entry_price = None
    
    def next(self):
        """交易逻辑"""
        if len(self.data) < self.slow_ma + 5:
            return
        
        current_price = self.data.Close[-1]
        
        # 检查指标有效性
        if np.isnan(self.sma_fast[-1]) or np.isnan(self.sma_medium[-1]) or np.isnan(self.sma_slow[-1]):
            return
        
        # 如果已有持仓，检查止损/止盈
        if self.position:
            self._check_exit(current_price)
            return
        
        try:
            # 双均线金叉 - 快速均线上穿中期均线，且中期上穿长期
            # 这样能过滤掉很多虚假信号
            if (crossover(self.sma_fast, self.sma_medium) and 
                self.sma_medium[-1] > self.sma_slow[-1]):
                # 多头信号 - 所有均线排列正确
                if self.sma_fast[-1] > self.sma_medium[-1] > self.sma_slow[-1]:
                    self.buy(size=self.position_size)
                    self.entry_price = current_price
            
            # 双均线死叉 - 快速均线下穿中期均线，且中期下穿长期
            elif (crossover(self.sma_medium, self.sma_fast) and 
                  self.sma_medium[-1] < self.sma_slow[-1]):
                # 空头信号
                if self.sma_fast[-1] < self.sma_medium[-1] < self.sma_slow[-1]:
                    self.sell(size=self.position_size)
                    self.entry_price = current_price
        except Exception:
            pass
    
    def _check_exit(self, current_price):
        """检查止损/止盈"""
        if not self.position or not self.entry_price:
            return
        
        try:
            if self.position.size > 0:
                # 多头止损/止盈
                if current_price <= self.entry_price * (1 - self.stop_loss_pct / 100):
                    self.position.close()
                elif current_price >= self.entry_price * (1 + self.take_profit_pct / 100):
                    self.position.close()
                # 当快速均线下穿中期均线时，也平仓
                elif crossover(self.sma_medium, self.sma_fast):
                    self.position.close()
            
            elif self.position.size < 0:
                # 空头止损/止盈
                if current_price >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                elif current_price <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                # 当快速均线上穿中期均线时，也平仓
                elif crossover(self.sma_fast, self.sma_medium):
                    self.position.close()
        except Exception:
            pass
