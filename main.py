from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import io
import base64
import json
from datetime import datetime
import warnings
import os
import tempfile
import importlib
import inspect

warnings.filterwarnings('ignore')

app = FastAPI(title="股票回测服务")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板目录
templates = Jinja2Templates(directory="templates")

# 定义请求模型
class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    interval: str = "1d"
    strategy_name: str = "sma_cross"
    initial_cash: float = 10000
    commission: float = 0.002

# 策略注册表 - 存储所有可用的策略
STRATEGY_REGISTRY = {}
STRATEGY_CONFIG = {}

# 加载策略配置
def load_strategy_config():
    """从配置文件加载策略配置"""
    config_path = "strategies.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"strategies": {}}

# 注册策略装饰器
def register_strategy(name, display_name=None, description=None):
    def decorator(strategy_class):
        STRATEGY_REGISTRY[name] = strategy_class
        # 保存策略配置信息
        STRATEGY_CONFIG[name] = {
            "name": name,
            "display_name": display_name or strategy_class.__name__,
            "description": description or getattr(strategy_class, "__doc__", "无描述"),
            "class_name": strategy_class.__name__
        }
        return strategy_class
    return decorator

# 动态加载策略模块
def load_strategy_modules():
    """动态加载所有策略模块"""
    strategies_dir = "strategies"
    if os.path.exists(strategies_dir) and os.path.isdir(strategies_dir):
        for file_name in os.listdir(strategies_dir):
            if file_name.endswith('.py') and file_name != '__init__.py':
                module_name = f"strategies.{file_name[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    # 注册模块中的所有策略类
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                            issubclass(obj, Strategy) and
                            obj != Strategy):
                            # 自动注册策略
                            strategy_name = name.lower()
                            register_strategy(strategy_name)(obj)
                except Exception as e:
                    print(f"加载策略模块 {module_name} 失败: {e}")


# 辅助指标函数
def RSI(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def BB_upper(series, period, std):
    return series.rolling(window=period).mean() + series.rolling(window=period).std() * std

def BB_lower(series, period, std):
    return series.rolling(window=period).mean() - series.rolling(window=period).std() * std

def Momentum(series, period):
    return series - series.shift(period)


# 定义简单的均线交叉策略
@register_strategy("sma_cross", "双均线交叉策略", "短期均线上穿长期均线时买入，下穿时卖出")
class SMACrossStrategy(Strategy):
    n1 = 10  # 短期均线周期
    n2 = 30  # 长期均线周期
    
    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)
    
    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

# RSI策略
@register_strategy("rsi", "RSI策略", "RSI低于30时买入（超卖），RSI高于70时卖出（超买）")
class RSIStrategy(Strategy):
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30
     
    def init(self):
        # self.I 会自动处理指标对齐和NaN值，确保策略的稳健性
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
    
    def next(self):
        # 只有在rsi值有效时才进行交易判断
        if not self.position:
            # 如果当前没有持仓，并且RSI低于阈值，则买入
            if self.rsi[-1] < self.rsi_lower:
                self.buy()
        else:
            # 如果当前有持仓，并且RSI高于阈值，则卖出
            if self.rsi[-1] > self.rsi_upper:
                self.sell()

# 布林带策略
@register_strategy("bollinger", "布林带策略", "价格跌破布林带下轨时买入，突破布林带上轨时卖出")
class BollingerBandsStrategy(Strategy):
    bb_period = 20
    bb_std = 2
    
    def init(self):
        close = self.data.Close
        self.bb_middle = self.I(SMA, close, self.bb_period)
        self.bb_upper = self.I(BB_upper, close, self.bb_period, self.bb_std)
        self.bb_lower = self.I(BB_lower, close, self.bb_period, self.bb_std)
    
    def next(self):
        if self.data.Close < self.bb_lower:
            self.buy()
        elif self.data.Close > self.bb_upper:
            self.sell()

# 动量策略
@register_strategy("momentum", "动量策略", "动量指标为正时买入，为负时卖出")
class MomentumStrategy(Strategy):
    momentum_period = 10
    
    def init(self):
        close = self.data.Close
        self.momentum = self.I(Momentum, close, self.momentum_period)
    
    def next(self):
        if self.momentum > 0:
            self.buy()
        elif self.momentum < 0:
            self.sell()

def prepare_data_for_backtesting(data):
    """准备数据以符合backtesting.py的格式要求"""
    # 确保列名首字母大写
    data.columns = [col.title() for col in data.columns]
    
    # 确保包含必要的列
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"数据缺少必要的列: {missing_columns}")
    
    # 确保数据是数值类型
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # 删除包含NaN值的行
    data = data.dropna()
    
    # 确保索引是日期时间格式
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # 按日期排序
    data = data.sort_index()
    
    return data

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    try:
        # 验证日期格式
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="开始日期必须早于结束日期")
        
        # 获取数据
        print(f"正在获取 {request.symbol} 的数据...")
        ticker = yf.Ticker(request.symbol)
        
        # 对于加密货币，可能需要特殊的处理
        if '-' in request.symbol and 'USD' in request.symbol:
            # 这是加密货币
            data = ticker.history(
                start=request.start_date,
                end=request.end_date,
                interval=request.interval
            )
        else:
            # 这是股票
            data = ticker.history(
                start=request.start_date,
                end=request.end_date,
                interval=request.interval
            )
        
        if data.empty:
            raise HTTPException(status_code=404, detail="无法获取数据，请检查股票代码和时间区间")
        
        print(f"原始数据形状: {data.shape}")
        print(f"原始数据列: {data.columns.tolist()}")
        
        # 准备数据
        data = prepare_data_for_backtesting(data)
        
        print(f"处理后数据形状: {data.shape}")
        print(f"处理后数据列: {data.columns.tolist()}")
        
        # 确保有足够的数据点
        if len(data) < 50:
            raise HTTPException(status_code=400, detail="数据点太少，无法进行有意义的回测")
        
        # 获取策略类
        strategy_class = STRATEGY_REGISTRY.get(request.strategy_name)
        if not strategy_class:
            raise HTTPException(status_code=400, detail=f"策略 '{request.strategy_name}' 不存在")
        
        # 执行回测
        print(f"正在执行回测，策略: {request.strategy_name}, 初始资金: {request.initial_cash}, 手续费: {request.commission}")
        bt = Backtest(data, strategy_class, cash=request.initial_cash, commission=request.commission)
        stats = bt.run()
        
        # 生成图表
        print("正在生成图表...")
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        try:
            # 生成图表并保存到临时文件
            bt.plot(filename=temp_filename, open_browser=False)
            
            # 读取HTML文件内容
            with open(temp_filename, 'r', encoding='utf-8') as f:
                plot_html_content = f.read()
            
            # 删除临时文件
            os.unlink(temp_filename)
            
        except Exception as e:
            # 确保删除临时文件
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            raise e
        
        print(stats)
        # 提取统计信息
        stats_dict = {
            "策略收益率": f"{stats['Return [%]']:.2f}%",
            "最大回撤": f"{stats['Max. Drawdown [%]']:.2f}%",
            "基准收益率": f"{stats['Buy & Hold Return [%]']:.2f}%",
            "持仓时间": f"{stats['Exposure Time [%]']:.2f}%",
            "年复合增长率": f"{stats['CAGR [%]']:.2f}%",        
            "交易次数": int(stats['# Trades']),
            "交易胜率": f"{stats['Win Rate [%]']:.2f}%",
            "夏普比率": f"{stats['Sharpe Ratio']:.2f}",
        }
        
        return {
            "plot_html": plot_html_content,
            "stats": stats_dict,
            "symbol": request.symbol,
            "interval": request.interval
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
    except Exception as e:
        print(f"详细错误信息: {str(e)}")
        raise HTTPException(status_code=500, detail=f"回测执行失败: {str(e)}")

@app.get("/strategies")
async def get_available_strategies():
    """获取所有可用的策略列表"""
    # 重新加载策略配置
    config = load_strategy_config()
    strategies = []
    
    # 合并配置文件和注册表中的策略
    for name in STRATEGY_REGISTRY:
        strategy_config = STRATEGY_CONFIG.get(name, {})
        strategies.append({
            "name": name,
            "display_name": strategy_config.get("display_name", name),
            "description": strategy_config.get("description", "无描述"),
            "class_name": strategy_config.get("class_name", "")
        })
    
    return JSONResponse(content=strategies)

@app.post("/reload-strategies")
async def reload_strategies():
    """重新加载所有策略"""
    try:
        # 清空当前注册表
        STRATEGY_REGISTRY.clear()
        STRATEGY_CONFIG.clear()
        
        # 重新加载配置
        config = load_strategy_config()
        
        # 重新加载动态策略模块
        load_strategy_modules()
        
        return {"message": "策略重新加载成功", "count": len(STRATEGY_REGISTRY)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新加载策略失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)