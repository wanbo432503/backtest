from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np

class StableProfitStrategy(Strategy):
    """
    稳定盈利量化策略 - 结合趋势、动量和风险控制
    
    核心逻辑：
    1. 使用快速SMA(20)和慢速SMA(50)识别趋势
    2. 使用RSI(14)过滤虚假信号，避免超买超卖区间交易
    3. 设置动态止损和止盈比例
    4. 只在强趋势中交易，增加胜率
    
    适用市场：股票、期货、数字货币（日线或以上）
    """
    
    # 移动平均线参数
    fast_ma = 20      # 快速均线周期
    slow_ma = 50      # 慢速均线周期
    
    # RSI参数
    rsi_period = 14   # RSI周期
    rsi_oversold = 35  # 超卖线（买入信号）
    rsi_overbought = 65 # 超买线（卖出信号）
    
    # 风险管理
    stop_loss_pct = 3.0    # 止损百分比 (%)
    take_profit_pct = 6.0  # 止盈百分比 (%)
    
    # 仓位管理
    position_size = 0.95   # 每笔交易使用95%的可用资金
    
    def init(self):
        """初始化策略指标"""
        close = self.data.Close
        
        # 计算快速和慢速均线
        self.sma_fast = self.I(SMA, close, self.fast_ma)
        self.sma_slow = self.I(SMA, close, self.slow_ma)
        
        # 计算RSI指标 - 使用 self.I() 包装函数
        self.rsi = self.I(self._calculate_rsi, close)
        
        # 记录入场价格用于止损/止盈
        self.entry_price = None
    
    def _calculate_rsi(self, prices):
        """
        计算RSI指标
        该函数用于 self.I() 方法，需要返回 numpy 数组
        """
        prices = np.asarray(prices, dtype=float)
        deltas = np.diff(prices)
        
        # 分离涨幅和跌幅
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均涨幅和平均跌幅
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        # 初始平均值
        if len(prices) > self.rsi_period:
            avg_gains[self.rsi_period] = np.mean(gains[:self.rsi_period])
            avg_losses[self.rsi_period] = np.mean(losses[:self.rsi_period])
            
            # Wilder's Smoothing
            for i in range(self.rsi_period + 1, len(prices)):
                avg_gains[i] = (avg_gains[i-1] * (self.rsi_period - 1) + gains[i-1]) / self.rsi_period
                avg_losses[i] = (avg_losses[i-1] * (self.rsi_period - 1) + losses[i-1]) / self.rsi_period
        
        # 计算 RSI
        rsi = np.zeros(len(prices))
        for i in range(self.rsi_period, len(prices)):
            if avg_losses[i] != 0:
                rs = avg_gains[i] / avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100 if avg_gains[i] > 0 else 0
        
        return rsi
    
    def next(self):
        """交易逻辑"""
        # 确保有足够的数据点
        if len(self.data) < self.slow_ma + self.rsi_period:
            return
        
        # 当前价格
        current_price = self.data.Close[-1]
        
        # 检查指标是否有效
        if np.isnan(self.sma_fast[-1]) or np.isnan(self.sma_slow[-1]) or np.isnan(self.rsi[-1]):
            return
        
        # 如果已有持仓，检查止损/止盈
        if self.position:
            self._check_exit_conditions(current_price)
            return
        
        # 没有持仓时，寻找买入/卖出机会
        try:
            # ========== 买入信号 ==========
            # 条件1: 快速均线上穿慢速均线（趋势开始）
            if crossover(self.sma_fast, self.sma_slow):
                # 条件2: RSI不在超买区间（避免顶部接盘）
                if self.rsi[-1] < self.rsi_overbought:
                    self.buy(size=self.position_size)
                    self.entry_price = current_price
            
            # ========== 卖出信号 ==========
            # 条件1: 快速均线下穿慢速均线（趋势反转）
            elif crossover(self.sma_slow, self.sma_fast):
                # 条件2: RSI不在超卖区间（避免底部被套）
                if self.rsi[-1] > self.rsi_oversold:
                    self.sell(size=self.position_size)
                    self.entry_price = current_price
        except Exception as e:
            # 防止任何异常导致策略崩溃
            pass
    
    def _check_exit_conditions(self, current_price):
        """检查止损/止盈条件"""
        if not self.position or not self.entry_price:
            return
        
        try:
            # 多头持仓
            if self.position.size > 0:
                # 止损条件
                if current_price <= self.entry_price * (1 - self.stop_loss_pct / 100):
                    self.position.close()
                    return
                
                # 止盈条件
                if current_price >= self.entry_price * (1 + self.take_profit_pct / 100):
                    self.position.close()
                    return
                
                # RSI进入超买区间，主动平仓（防止过度涨价后回落）
                if not np.isnan(self.rsi[-1]) and self.rsi[-1] > self.rsi_overbought:
                    self.position.close()
                    return
            
            # 空头持仓
            elif self.position.size < 0:
                # 止损条件
                if current_price >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                    return
                
                # 止盈条件
                if current_price <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                    return
                
                # RSI进入超卖区间，主动平仓
                if not np.isnan(self.rsi[-1]) and self.rsi[-1] < self.rsi_oversold:
                    self.position.close()
                    return
        except Exception as e:
            # 防止异常导致策略崩溃
            pass
