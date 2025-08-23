# 股票回测系统

基于FastAPI和backtesting.py的股票回测系统，支持多种交易策略和动态策略加载。

## 功能特性

- ✅ 多种内置回测策略（均线交叉、RSI、布林带、动量）
- ✅ 可配置初始资金和交易手续费
- ✅ 动态策略加载机制
- ✅ 美观的Web界面
- ✅ 实时策略刷新功能

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python main.py
```

访问 http://localhost:8000 使用回测系统。

## 策略管理

### 内置策略

系统内置了以下策略：
- `sma_cross` - 双均线交叉策略
- `rsi` - RSI策略  
- `bollinger` - 布林带策略
- `momentum` - 动量策略

### 添加自定义策略

1. 在 `strategies/` 目录下创建新的策略文件（如 `my_strategy.py`）
2. 策略类必须继承 `backtesting.Strategy`
3. 文件保存后，点击前端的"刷新策略列表"按钮即可加载新策略

### 示例策略文件

```python
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

class MyCustomStrategy(Strategy):
    """自定义策略说明"""
    
    # 策略参数
    param1 = 10
    param2 = 20
    
    def init(self):
        # 初始化指标
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.param1)
        self.sma2 = self.I(SMA, close, self.param2)
    
    def next(self):
        # 交易逻辑
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()
```

### 动态刷新策略

系统支持动态加载策略，无需重启服务：
1. 添加新的策略文件到 `strategies/` 目录
2. 在前端点击"刷新策略列表"按钮
3. 新策略会自动出现在策略选择列表中

## API接口

### 获取策略列表
```
GET /strategies
```

### 重新加载策略
```
POST /reload-strategies
```

### 执行回测
```
POST /backtest
{
    "symbol": "AAPL",
    "strategy_name": "sma_cross",
    "initial_cash": 10000,
    "commission": 0.002,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "interval": "1d"
}
```

## 配置说明

策略配置保存在 `strategies.json` 文件中，可以手动编辑该文件来修改策略的显示名称和描述。

## 注意事项

- 分钟级数据最多支持7天回测
- 小时级数据最多支持730天回测  
- 确保策略文件没有语法错误，否则加载会失败
- 自定义策略类名不能与现有策略重复