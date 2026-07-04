# 股票回测系统

Backtest 是一个面向 A 股代码的量化回测 Web 应用，基于 FastAPI、Backtesting.py 和免费公开数据源构建。当前产品口径只支持 A 股代码，不再支持美股、港股或加密货币作为回测对象。

## 核心功能

- A 股代码回测：支持 6 位代码、`SH/SZ/BJ` 前缀代码，以及 `.SH/.SZ/.BJ` 后缀代码。
- 策略回测：支持均线、RSI、布林带、动量等策略，通过策略注册表动态加载。
- 数据源选择：`auto` 默认优先使用 `mootdx`，失败后回退到 `yfinance` 的 A 股后缀代码。
- A 股搜索：支持通过 A 股代码和中文公司名称搜索，搜索结果统一回填 A 股代码。
- 信息面板：展示研报、资金、龙虎榜、公告等 A 股辅助信息。
- Web 页面：左侧参数设置，中间统计和图表，右侧标的信息。

## 启动

```bash
pip install -r requirements.txt
python main.py
```

服务默认运行在 `http://localhost:8005`。

## 数据源

- `mootdx`：A 股 K 线主数据源，默认优先使用。
- `yfinance`：仅作为 A 股代码转换后的备用 K 线来源。
- 腾讯财经：用于 A 股实时行情和估值信息。
- 东方财富：用于研报、资金流、龙虎榜等信息面板。
- 巨潮资讯：用于公告信息。

注意：`mootdx` 返回不复权原始价，跨除权除息日回测需要谨慎解释结果。

## A 股代码格式

支持以下输入：

- `SZ002241`
- `SH600519`
- `BJ430047`
- `002241`
- `600519.SH`
- `000001.SZ`

不支持以下输入：

- `AAPL`
- `MSFT`
- `0700.HK`
- `BTC-USD`

## 常用 API

搜索 A 股：

```bash
curl "http://localhost:8005/search-stocks?query=中科曙光"
```

执行回测：

```bash
curl -X POST "http://localhost:8005/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SZ002241",
    "start_date": "2025-07-03",
    "end_date": "2026-07-04",
    "interval": "1h",
    "strategy_name": "rsi",
    "initial_cash": 10000,
    "commission": 0.002,
    "data_provider": "auto"
  }'
```

获取信息面板：

```bash
curl "http://localhost:8005/market-insights/SZ002241"
```

## 测试

```bash
python -m pytest test/test_market_data_sources.py test/test_stock_search_cn.py test/test_market_insights_api.py test/test_index_template.py -q
python -m py_compile market_data.py stock_search.py main.py
```

## 开发约定

- 保持回测对象聚焦 A 股代码。
- 修改数据源、搜索或 UI 后运行聚焦测试。
- 完成并验证一个完整改动后及时提交。
