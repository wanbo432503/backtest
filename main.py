from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from backtesting import Strategy
import io
import base64
import json
import warnings
import os
import importlib
import inspect
from stock_search import search_stocks
from market_data import normalize_symbol
from optimization_models import AShareTradingConfig, OptimizationRequest, RiskConfig
from portfolio_backtest_runner import run_portfolio_backtest
from portfolio_models import PortfolioBacktestRequest
from portfolio_progress import PortfolioBacktestJobStore
from strategy_metadata import get_strategy_parameters
from backtest_runner import run_single_backtest
from optimization_runner import run_optimization
from universe_scan_runner import run_universe_scan
from tradingagents_adapter import (
    TradingAgentsAdapterError,
    run_tradingagents_analysis,
    run_tradingagents_portfolio_summary,
)
from tradingagents_config import (
    get_config_api_key as get_tradingagents_config_api_key,
    get_config_view as get_tradingagents_config_view,
    test_config as test_tradingagents_config,
    update_config as update_tradingagents_config,
)
from tradingagents_models import (
    TradingAgentsAnalysisRequest,
    TradingAgentsConfigUpdate,
    TradingAgentsPortfolioSummaryRequest,
)
from tradable_universe import validate_universe

warnings.filterwarnings('ignore')

app = FastAPI(title="股票回测服务")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板目录
templates = Jinja2Templates(directory="templates")
portfolio_job_store = PortfolioBacktestJobStore(run_portfolio_backtest)

# 定义请求模型
class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    interval: str = "1d"
    strategy_name: str = "macd_volume_divergence_risk_control"
    strategy_params: dict = Field(default_factory=dict)
    initial_cash: float = 10000
    commission: float = 0.002
    data_provider: str = "auto"
    risk_config: RiskConfig | None = None
    a_share_config: AShareTradingConfig | None = None


class UniverseValidationRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)

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

load_strategy_modules()


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
            strategy_params=request.strategy_params,
        )
        return result.to_api_response()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
    except Exception as e:
        print(f"详细错误信息: {str(e)}")
        raise HTTPException(status_code=500, detail=f"回测执行失败: {str(e)}")

@app.post("/optimize")
async def optimize_endpoint(payload: dict):
    """运行单只 A 股标的的参数优化。"""
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


@app.post("/portfolio/validate-universe")
async def validate_portfolio_universe(request: UniverseValidationRequest):
    result = validate_universe(request.symbols)
    return {
        "ok": result.ok,
        "accepted_symbols": result.accepted_symbols,
        "rejected": [
            {
                "raw": row.raw,
                "symbol": row.symbol,
                "normalized_symbol": row.normalized_symbol,
                "reason": row.reason,
            }
            for row in result.rejected
        ],
    }


@app.post("/portfolio-backtest")
async def portfolio_backtest_endpoint(payload: dict):
    try:
        request = PortfolioBacktestRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = await run_in_threadpool(run_portfolio_backtest, request)
        return result.to_api_response()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"组合回测详细错误信息: {str(e)}")
        raise HTTPException(status_code=500, detail=f"组合回测执行失败: {str(e)}")


@app.post("/portfolio-backtest/jobs")
async def create_portfolio_backtest_job(payload: dict):
    try:
        request = PortfolioBacktestRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    snapshot = portfolio_job_store.submit(request)
    return snapshot.to_api_response()


@app.get("/portfolio-backtest/jobs/{job_id}")
async def get_portfolio_backtest_job(job_id: str):
    snapshot = portfolio_job_store.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="portfolio backtest job not found")
    return snapshot.to_api_response()


@app.post("/portfolio/universe-scan")
async def portfolio_universe_scan_endpoint(payload: dict):
    try:
        request = PortfolioBacktestRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = await run_in_threadpool(run_universe_scan, request)
        return result.to_api_response()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"股票池扫描详细错误信息: {str(e)}")
        raise HTTPException(status_code=500, detail=f"股票池扫描失败: {str(e)}")

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


@app.get("/tradingagents/config")
async def get_tradingagents_config_endpoint():
    try:
        return get_tradingagents_config_view()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 TradingAgents 配置失败: {str(e)}")


@app.get("/tradingagents/config/api-key")
async def get_tradingagents_config_api_key_endpoint():
    try:
        return get_tradingagents_config_api_key()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 TradingAgents API Key 失败: {str(e)}")


@app.put("/tradingagents/config")
async def update_tradingagents_config_endpoint(payload: TradingAgentsConfigUpdate):
    try:
        return update_tradingagents_config(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存 TradingAgents 配置失败: {str(e)}")


@app.post("/tradingagents/config/test")
async def test_tradingagents_config_endpoint():
    try:
        return test_tradingagents_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检查 TradingAgents 配置失败: {str(e)}")


@app.post("/tradingagents/analysis")
async def run_tradingagents_analysis_endpoint(payload: TradingAgentsAnalysisRequest):
    try:
        return await run_in_threadpool(run_tradingagents_analysis, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TradingAgentsAdapterError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TradingAgents 分析失败: {str(e)}")


@app.post("/tradingagents/portfolio-summary")
async def run_tradingagents_portfolio_summary_endpoint(payload: dict):
    try:
        request = TradingAgentsPortfolioSummaryRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        return await run_in_threadpool(run_tradingagents_portfolio_summary, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TradingAgentsAdapterError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TradingAgents 组合总结失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
