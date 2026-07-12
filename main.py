from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError
import warnings
from stock_search import search_stocks
from market_data import normalize_symbol
from optimization_models import AShareTradingConfig, OptimizationRequest, RiskConfig
from optimization_progress import OptimizationJobStore
from portfolio_backtest_runner import run_portfolio_backtest
from portfolio_factor_optimization_models import PortfolioFactorOptimizationRequest
from portfolio_factor_optimization_progress import PortfolioFactorOptimizationJobStore
from portfolio_factor_optimizer import run_factor_optimization
from portfolio_models import PortfolioBacktestRequest
from portfolio_selection_strategy_library import list_selection_strategies
from portfolio_progress import PortfolioBacktestJobStore
from signal_portfolio_models import SignalPortfolioBacktestRequest
from signal_portfolio_runner import run_signal_portfolio_backtest
from strategy_library import get_strategy_library, reload_strategy_library
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
signal_portfolio_job_store = PortfolioBacktestJobStore(
    run_signal_portfolio_backtest,
    job_label="多股票信号组合回测",
)
portfolio_factor_optimization_job_store = PortfolioFactorOptimizationJobStore(run_factor_optimization)

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

STRATEGY_LIBRARY = get_strategy_library()
optimization_job_store = OptimizationJobStore(
    run_optimization,
    strategy_library=STRATEGY_LIBRARY,
)


def load_strategy_modules():
    """Compatibility entry point that atomically rebuilds the unified library."""
    global STRATEGY_LIBRARY
    STRATEGY_LIBRARY = reload_strategy_library()
    optimization_job_store._strategy_library = STRATEGY_LIBRARY
    return STRATEGY_LIBRARY


@app.on_event("startup")
def startup_event():
    """Validate and refresh the unified strategy library on startup."""
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
            strategy_library=STRATEGY_LIBRARY,
            initial_cash=request.initial_cash,
            commission=request.commission,
            data_provider=request.data_provider,
            strategy_params=request.strategy_params,
            trading_config=request.a_share_config,
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
    request = _validate_optimization_request(payload)

    try:
        result = run_optimization(request, strategy_library=STRATEGY_LIBRARY)
        return result.to_api_response()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"优化参数错误: {str(e)}")
    except Exception as e:
        print(f"优化详细错误信息: {str(e)}")
        raise HTTPException(status_code=500, detail=f"优化执行失败: {str(e)}")


@app.post("/optimization/jobs")
async def create_optimization_job(payload: dict):
    request = _validate_optimization_request(payload)
    snapshot = optimization_job_store.submit(request)
    return snapshot.to_api_response()


@app.get("/optimization/jobs/{job_id}")
async def get_optimization_job(job_id: str):
    snapshot = optimization_job_store.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="optimization job not found")
    return snapshot.to_api_response()


def _validate_optimization_request(payload: dict) -> OptimizationRequest:
    try:
        request = OptimizationRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    for symbol in request.optimization_config.symbols:
        if normalize_symbol(symbol).market != "CN":
            raise HTTPException(status_code=400, detail=f"仅支持 A 股代码: {symbol}")
    return request


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


@app.post("/signal-portfolio-backtest/jobs")
async def create_signal_portfolio_backtest_job(payload: dict):
    try:
        request = SignalPortfolioBacktestRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    snapshot = signal_portfolio_job_store.submit(request)
    return snapshot.to_api_response()


@app.get("/signal-portfolio-backtest/jobs/{job_id}")
async def get_signal_portfolio_backtest_job(job_id: str):
    snapshot = signal_portfolio_job_store.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="signal portfolio backtest job not found")
    return snapshot.to_api_response()


@app.post("/portfolio-factor-optimization/jobs")
async def create_portfolio_factor_optimization_job(payload: dict):
    try:
        request = PortfolioFactorOptimizationRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    snapshot = portfolio_factor_optimization_job_store.submit(request)
    return snapshot.to_api_response()


@app.get("/portfolio-factor-optimization/jobs/{job_id}")
async def get_portfolio_factor_optimization_job(job_id: str):
    snapshot = portfolio_factor_optimization_job_store.get(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="portfolio factor optimization job not found")
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


@app.get("/portfolio-selection-strategies")
async def portfolio_selection_strategies_endpoint():
    return {
        "strategies": [
            strategy.model_dump(mode="json")
            for strategy in list_selection_strategies()
        ]
    }


@app.get("/strategies")
async def get_available_strategies():
    """获取统一单股/组合策略目录。"""
    return STRATEGY_LIBRARY.to_catalog()

@app.post("/reload-strategies")
async def reload_strategies():
    """Atomically rebuild and validate the unified strategy library."""
    try:
        library = load_strategy_modules()
        return {"message": "策略重新加载成功", "count": len(library.list())}
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
