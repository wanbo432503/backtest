from backtesting import Strategy
import numpy as np

class BreakthroughMomentumStrategy(Strategy):
    """
    突破动量策略 - 捕捉价格大幅上升的机会
    
    特点：
    1. 基于价格突破和动量指标
    2. 只在强势上升或下跌时交易
    3. 较大的利润目标，追求大幅收益
    4. 适合波动较大的市场
    
    逻辑：
    - 如果最近 N 天的涨幅很大，且还在上升趋势中，则买入
    - 基于高低点的突破进出场
    """
    
    # 参数
    lookback_period = 10      # 回看周期
    momentum_threshold = 0.02  # 动量阈值（2%）
    
    # 风险管理
    stop_loss_pct = 2.5       # 2.5% 止损
    take_profit_pct = 8.0     # 8% 止盈（追求大幅收益）
    
    position_size = 0.95
    
    def init(self):
        """初始化策略指标"""
        self.entry_price = None
    
    def _calculate_momentum(self):
        """计算最近 N 天的价格动量"""
        if len(self.data) < self.lookback_period:
            return 0
        
        current_price = self.data.Close[-1]
        past_price = self.data.Close[-self.lookback_period]
        
        # 计算百分比变化
        momentum = (current_price - past_price) / past_price
        return momentum
    
    def _calculate_highest_low(self):
        """获取最近 N 天的最低点"""
        if len(self.data) < self.lookback_period:
            return self.data.Low[-1]
        return np.min(self.data.Low[-self.lookback_period:])
    
    def _calculate_lowest_high(self):
        """获取最近 N 天的最高点"""
        if len(self.data) < self.lookback_period:
            return self.data.High[-1]
        return np.max(self.data.High[-self.lookback_period:])
    
    def next(self):
        """交易逻辑"""
        if len(self.data) < self.lookback_period + 5:
            return
        
        current_price = self.data.Close[-1]
        momentum = self._calculate_momentum()
        
        # 如果已有持仓，检查止损/止盈
        if self.position:
            self._check_exit(current_price)
            return
        
        try:
            # 买入信号：价格在上升趋势中，且最近动量为正且较强
            if momentum > self.momentum_threshold:
                # 额外确认：当前价格应该接近最近的高点
                highest_point = self._calculate_lowest_high()
                if current_price > highest_point * 0.98:  # 在高点附近
                    self.buy(size=self.position_size)
                    self.entry_price = current_price
            
            # 卖出信号：价格在下降趋势中，且最近动量为负
            elif momentum < -self.momentum_threshold:
                # 额外确认：当前价格应该接近最近的低点
                lowest_point = self._calculate_highest_low()
                if current_price < lowest_point * 1.02:  # 在低点附近
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
                # 动量反转时也平仓
                elif self._calculate_momentum() < 0:
                    self.position.close()
            
            elif self.position.size < 0:
                # 空头止损/止盈
                if current_price >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                elif current_price <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                # 动量反转时也平仓
                elif self._calculate_momentum() > 0:
                    self.position.close()
        except Exception:
            pass
