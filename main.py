from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import io
import base64
import json
import warnings
import os
import importlib
import inspect
from stock_search import search_stocks, get_stock_info
from market_data import normalize_symbol
from market_insights import get_market_insights
from optimization_models import AShareTradingConfig, OptimizationRequest, RiskConfig
from strategy_metadata import get_strategy_parameters
from backtest_runner import run_single_backtest
from optimization_runner import run_optimization

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
    data_provider: str = "auto"
    risk_config: RiskConfig | None = None
    a_share_config: AShareTradingConfig | None = None

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
                            strategy_name = getattr(obj, "strategy_name", name.lower())
                            display_name = getattr(obj, "display_name", None)
                            description = getattr(obj, "description", None)
                            register_strategy(strategy_name, display_name, description)(obj)
                except Exception as e:
                    print(f"加载策略模块 {module_name} 失败: {e}")

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
            self.position.close()

def RSI(array, n):
    """Relative strength index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)

# RSI策略
@register_strategy("rsi", "RSI策略", "RSI低于30时买入（超卖），RSI高于70时卖出（超买）")
class RSIStrategy(Strategy):
    # 定义策略参数
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30

    def init(self):
        """初始化策略，计算指标"""
        print(f"初始化RSI策略，周期: {self.rsi_period}, 上阈值: {self.rsi_upper}, 下阈值: {self.rsi_lower}")
        
        # 使用 self.I() 调用独立的RSI计算函数
        # 第一个参数是函数名，后续参数是传递给该函数的数据
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        
        print("RSI指标初始化完成")

    def next(self):
        """定义交易逻辑"""
        # 在 backtesting.py 中，通常使用 crossover 函数来判断穿越条件，这样更精确
        # 也可以直接比较当前RSI值
        current_rsi = self.rsi[-1]
        
        # 如果当前没有持仓，并且RSI从下方上穿了下阈值
        if crossover(self.rsi, self.rsi_lower):
            if not self.position:
                print(f"🚀 买入信号: RSI({current_rsi:.2f}) 穿越 {self.rsi_lower}")
                self.buy()
        # 如果当前有持仓，并且RSI从下方上穿了上阈值
        elif crossover(self.rsi, self.rsi_upper):
            if self.position:
                print(f"💰 卖出信号: RSI({current_rsi:.2f}) 穿越 {self.rsi_upper}")
                self.position.close()


# 计算布林带的函数
def bollinger_bands(price_series, n, k):
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

# 布林带策略
@register_strategy("bollinger", "布林带策略", "价格跌破布林带下轨时买入，突破布林带上轨时卖出")
class BollingerBandsStrategy(Strategy):
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
                print(f"🚀 买入信号: 价格跌破布林下轨 {lower_band}")
                self.buy()
        # 卖出逻辑：当价格从下方向上穿过（突破）上轨时
        elif crossover(self.data.Close, upper_band):
            # 如果当前有持仓，则平仓卖出
            if self.position:
                print(f"💰 卖出信号: 价格上穿布林上轨 {upper_band}")
                self.position.close()

    
def momentum_indicator(array, n):
    """
    计算动量指标
    :param array: 价格序列 (例如收盘价)
    :param n: 计算动量的时间窗口
    :return: 动量值的序列
    """
    # 动量 = 当前价格 - n周期前的价格
    return pd.Series(array).diff(n)

# 动量策略
@register_strategy("momentum", "动量策略", "动量指标为正时买入，为负时卖出")
class MomentumStrategy(Strategy):
    """
    动量策略：
    - 当动量指标为正时买入
    - 当动量指标为负时卖出
    """
    # 定义一个参数，用于优化动量的时间窗口
    momentum_window = 20

    def init(self):
        # 在策略初始化时，计算动量指标
        # self.I() 函数用于将一个函数或指标应用于策略的数据
        self.momentum = self.I(momentum_indicator, self.data.Close, self.momentum_window)

    def next(self):
        # next() 方法会在每个数据点（例如每天）被调用，用于执行交易逻辑

        # 如果动量为正，并且我们当前没有持仓，则买入
        if self.momentum > 0 and not self.position.is_long:
            self.buy()
        # 如果动量变为负，并且我们当前持有多头头寸，则平仓
        elif self.momentum < 0 and self.position.is_long:
            self.position.close()

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

@app.on_event("startup")
def startup_event():
    """应用启动时执行"""
    load_strategy_modules()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    try:
        result = run_single_backtest(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
            strategy_name=request.strategy_name,
            strategy_registry=STRATEGY_REGISTRY,
            initial_cash=request.initial_cash,
            commission=request.commission,
            data_provider=request.data_provider,
        )
        return result.to_api_response()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
    except Exception as e:
        print(f"详细错误信息: {str(e)}")
        raise HTTPException(status_code=500, detail=f"回测执行失败: {str(e)}")

@app.post("/optimize")
async def optimize_endpoint(payload: dict):
    """运行固定 A 股股票池的参数优化。"""
    try:
        request = OptimizationRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    for symbol in request.optimization_config.symbols:
        if normalize_symbol(symbol).market != "CN":
            raise HTTPException(status_code=400, detail=f"仅支持 A 股代码: {symbol}")

    try:
        result = run_optimization(request, strategy_registry=STRATEGY_REGISTRY)
        return result.to_api_response()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"优化参数错误: {str(e)}")
    except Exception as e:
        print(f"优化详细错误信息: {str(e)}")
        raise HTTPException(status_code=500, detail=f"优化执行失败: {str(e)}")

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
            "class_name": strategy_config.get("class_name", ""),
            "parameters": get_strategy_parameters(name),
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

@app.get("/search-stocks")
async def search_stocks_endpoint(query: str = None):
    """
    搜索 A 股代码
    参数: query - A 股公司名字或股票代码
    返回: 匹配的 A 股列表
    """
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="搜索关键词不能为空")
    
    try:
        results = search_stocks(query.strip(), market="CN")
        
        if not results:
            raise HTTPException(status_code=404, detail=f"未找到与 '{query}' 相关的股票")
        
        return {
            "query": query,
            "count": len(results),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.get("/search-stocks-multi-market")
async def search_stocks_multi_market(query: str = None, market: str = None):
    """
    搜索 A 股代码
    参数: 
        - query: 公司名字或股票代码
        - market: 仅支持 CN/A股
    返回: 匹配的 A 股列表（包含市场信息）
    """
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="搜索关键词不能为空")
    
    # 标准化市场参数
    market_map = {
        'cn': 'CN', 'a股': 'CN', 'a': 'CN',
    }
    
    target_market = None
    if market:
        market_lower = market.lower()
        target_market = market_map.get(market_lower, market.upper() if len(market) <= 2 else None)
        if target_market != 'CN':
            raise HTTPException(status_code=400, detail=f"不支持的市场: {market}，当前仅支持 CN/A股")
    else:
        target_market = "CN"
    
    try:
        from stock_search import search_stocks as search_fn
        results = search_fn(query.strip(), market=target_market)
        
        if not results:
            raise HTTPException(status_code=404, detail=f"未找到与 '{query}' 相关的A股")
        
        # 为结果添加市场中文名称
        market_names = {'CN': 'A股'}
        for result in results:
            result['market_name'] = market_names.get(result.get('market'), result.get('market', '未知'))
        
        return {
            "query": query,
            "market": target_market,
            "count": len(results),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.get("/search-us-stocks")
async def search_us_stocks(query: str = None):
    """搜索美股"""
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="搜索关键词不能为空")
    
    try:
        from stock_search import search_us_stocks
        results = search_us_stocks(query.strip())
        
        if not results:
            raise HTTPException(status_code=404, detail=f"未找到与 '{query}' 相关的美股")
        
        return {
            "query": query,
            "market": "美股 (US)",
            "count": len(results),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.get("/search-cn-stocks")
async def search_cn_stocks(query: str = None):
    """搜索A股"""
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="搜索关键词不能为空")
    
    try:
        from stock_search import search_cn_stocks
        results = search_cn_stocks(query.strip())
        
        if not results:
            raise HTTPException(status_code=404, detail=f"未找到与 '{query}' 相关的A股")
        
        return {
            "query": query,
            "market": "A股 (CN)",
            "count": len(results),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.get("/search-hk-stocks")
async def search_hk_stocks(query: str = None):
    """搜索港股"""
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="搜索关键词不能为空")
    
    try:
        from stock_search import search_hk_stocks
        results = search_hk_stocks(query.strip())
        
        if not results:
            raise HTTPException(status_code=404, detail=f"未找到与 '{query}' 相关的港股")
        
        return {
            "query": query,
            "market": "港股 (HK)",
            "count": len(results),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.get("/stock-info/{symbol}")
async def get_stock_info_endpoint(symbol: str):
    """
    获取股票详细信息
    参数: symbol - 股票代码
    """
    if not symbol or len(symbol.strip()) == 0:
        raise HTTPException(status_code=400, detail="股票代码不能为空")
    
    try:
        info = get_stock_info(symbol.upper().strip())
        
        if 'error' in info:
            raise HTTPException(status_code=404, detail=info['error'])
        
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票信息失败: {str(e)}")

@app.get("/market-insights/{symbol}")
async def market_insights_endpoint(symbol: str):
    """获取右侧标的信息面板数据"""
    if not symbol or len(symbol.strip()) == 0:
        raise HTTPException(status_code=400, detail="股票代码不能为空")

    try:
        return get_market_insights(symbol.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取市场信息失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
