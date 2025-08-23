from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

# 定义简单的均线交叉策略
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
        
        # 执行回测
        print("正在执行回测...")
        bt = Backtest(data, SMACrossStrategy, cash=10000, commission=0.002)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)