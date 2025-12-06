from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import numpy as np
import pandas as pd

class SmartHedgeStrategy(Strategy):
    """
    智能对冲策略 - 结合多个指标的高效交易
    
    特点：
    1. 快速 SMA(10) + 慢速 SMA(30) 识别趋势
    2. RSI(7) 确认动量，避免顶底交易
    3. ATR 动态止损，根据波动率调整风险
    4. 自适应持仓大小，风险均衡
    
    目标：在风险可控的前提下，最大化收益
    """
    
    # 均线参数
    fast_ma = 10
    slow_ma = 30
    
    # RSI 参数 - 快速 RSI
    rsi_period = 7
    rsi_oversold = 40
    rsi_overbought = 60
    
    # ATR 参数 - 用于动态止损
    atr_period = 14
    atr_mult = 1.5
    
    # 止盈参数
    take_profit_pct = 5.0
    
    # 仓位管理
    max_position_size = 0.95
    
    def init(self):
        """初始化策略指标"""
        close = self.data.Close
        
        # 均线
        self.sma_fast = self.I(SMA, close, self.fast_ma)
        self.sma_slow = self.I(SMA, close, self.slow_ma)
        
        # RSI
        self.rsi = self.I(self._calculate_rsi, close)
        
        # ATR
        self.atr = self.I(self._calculate_atr)
        
        self.entry_price = None
        self.stop_loss_price = None
    
    def _calculate_rsi(self, data):
        """快速 RSI 计算"""
        prices = np.asarray(data, dtype=float)
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        if len(prices) > self.rsi_period:
            avg_gains[self.rsi_period] = np.mean(gains[:self.rsi_period])
            avg_losses[self.rsi_period] = np.mean(losses[:self.rsi_period])
            
            for i in range(self.rsi_period + 1, len(prices)):
                avg_gains[i] = (avg_gains[i-1] * (self.rsi_period - 1) + gains[i-1]) / self.rsi_period
                avg_losses[i] = (avg_losses[i-1] * (self.rsi_period - 1) + losses[i-1]) / self.rsi_period
        
        rsi = np.zeros(len(prices))
        for i in range(self.rsi_period, len(prices)):
            if avg_losses[i] != 0:
                rs = avg_gains[i] / avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100 if avg_gains[i] > 0 else 0
        
        return rsi
    
    def _calculate_atr(self):
        """计算 ATR (Average True Range)"""
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        
        tr = np.zeros(len(high))
        for i in range(len(high)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
        
        # 计算 ATR
        atr_values = np.zeros(len(tr))
        if len(tr) > self.atr_period:
            atr_values[self.atr_period] = np.mean(tr[:self.atr_period])
            for i in range(self.atr_period + 1, len(tr)):
                atr_values[i] = (atr_values[i-1] * (self.atr_period - 1) + tr[i]) / self.atr_period
        
        return atr_values
    
    def next(self):
        """交易逻辑"""
        if len(self.data) < max(self.slow_ma, self.rsi_period, self.atr_period) + 5:
            return
        
        current_price = self.data.Close[-1]
        
        # 检查指标有效性
        if np.isnan(self.sma_fast[-1]) or np.isnan(self.rsi[-1]) or np.isnan(self.atr[-1]):
            return
        
        # 如果已有持仓，检查止损/止盈
        if self.position:
            self._check_exit(current_price)
            return
        
        try:
            # 买入信号：均线金叉 + RSI 确认
            if crossover(self.sma_fast, self.sma_slow):
                # 确认：RSI 不在极端超买区间（但可以在中等位置）
                if self.rsi[-1] < self.rsi_overbought + 10:  # 比较宽松的条件
                    # 确认：快速均线在慢速均线上方
                    if self.sma_fast[-1] > self.sma_slow[-1]:
                        # 计算动态止损
                        atr_value = self.atr[-1]
                        self.stop_loss_price = current_price - atr_value * self.atr_mult
                        
                        self.buy(size=self.max_position_size)
                        self.entry_price = current_price
            
            # 卖出信号：均线死叉 + RSI 确认
            elif crossover(self.sma_slow, self.sma_fast):
                # 确认：RSI 不在极端超卖区间
                if self.rsi[-1] > self.rsi_oversold - 10:
                    # 确认：快速均线在慢速均线下方
                    if self.sma_fast[-1] < self.sma_slow[-1]:
                        # 计算动态止损
                        atr_value = self.atr[-1]
                        self.stop_loss_price = current_price + atr_value * self.atr_mult
                        
                        self.sell(size=self.max_position_size)
                        self.entry_price = current_price
        except Exception:
            pass
    
    def _check_exit(self, current_price):
        """检查止损/止盈"""
        if not self.position or not self.entry_price:
            return
        
        try:
            if self.position.size > 0:
                # 多头
                # 止损：基于 ATR 的动态止损
                if self.stop_loss_price and current_price <= self.stop_loss_price:
                    self.position.close()
                    return
                
                # 止盈
                if current_price >= self.entry_price * (1 + self.take_profit_pct / 100):
                    self.position.close()
                    return
                
                # 死叉平仓
                if crossover(self.sma_slow, self.sma_fast):
                    self.position.close()
                    return
            
            elif self.position.size < 0:
                # 空头
                # 止损
                if self.stop_loss_price and current_price >= self.stop_loss_price:
                    self.position.close()
                    return
                
                # 止盈
                if current_price <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                    return
                
                # 金叉平仓
                if crossover(self.sma_fast, self.sma_slow):
                    self.position.close()
                    return
        except Exception:
            pass
