# 股票回测系统

Backtest 是一个面向 A 股代码的量化回测 Web 应用，基于 FastAPI、Backtesting.py 和免费公开数据源构建。当前产品口径只支持 A 股代码，不再支持美股、港股或加密货币作为回测对象。

## 核心功能

- A 股代码回测：支持 6 位代码、`SH/SZ/BJ` 前缀代码，以及 `.SH/.SZ/.BJ` 后缀代码。
- 策略回测：支持均线、RSI、布林带、动量等策略，通过策略注册表动态加载。
- 数据源选择：`auto` 默认优先使用 `mootdx`，失败后回退到 `yfinance` 的 A 股后缀代码。
- A 股搜索：支持通过 A 股代码和中文公司名称搜索，搜索结果统一回填 A 股代码。
- 信息面板：展示研报、资金、龙虎榜、公告等 A 股辅助信息。
- Phase 2.0 策略优化：支持固定 1-3 只 A 股标的，按 `score` 对策略参数组合排序。
- 风控与 A 股规则：支持止损、止盈、仓位比例、最大持仓周期、T+1、100 股整数手、涨跌停过滤、滑点、佣金、印花税等配置。
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

执行参数优化：

```bash
curl -X POST "http://localhost:8005/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2025-07-03",
    "end_date": "2026-07-04",
    "interval": "1d",
    "data_provider": "auto",
    "optimization_config": {
      "symbols": ["SH603019", "SZ002241"],
      "objective": "score",
      "top_n": 10,
      "max_combinations": 300,
      "min_trades": 5,
      "train_start_date": "2025-07-03",
      "train_end_date": "2025-12-31",
      "validate_start_date": "2026-01-01",
      "validate_end_date": "2026-07-04",
      "strategies": [
        {
          "strategy_name": "rsi_risk_control",
          "fixed_params": {"trend_ma": 60},
          "search_space": {
            "rsi_period": [6, 14, 21],
            "rsi_buy": [25, 30, 35],
            "rsi_sell": [60, 70, 80]
          }
        }
      ]
    }
  }'
```

获取信息面板：

```bash
curl "http://localhost:8005/market-insights/SZ002241"
```

## Phase 2.0 优化工作流

Phase 2.0 面向固定少数 A 股标的的集中投资场景，而不是多股票组合轮动。默认股票池建议从中科曙光 `SH603019` 和歌尔股份 `SZ002241` 开始，也可以额外加入第三只 A 股。

推荐流程：

1. 选择固定股票池。
2. 选择策略，例如 RSI 风控、均线趋势风控或放量突破风控。
3. 设置风控参数和 A 股交易规则。
4. 启用参数优化，配置 Top N、最大组合数和最小交易次数。
5. 使用训练/验证切分查看 Top 结果。
6. 对最佳参数点击“应用并回测”，再看单次图表细节。

## Score 口径

优化目标固定为 `score`：

```text
score = 年化收益率 * 0.4 + 夏普比率 * 0.3 - 最大回撤绝对值 * 0.3
```

字段口径：

- 年化收益率：百分比数值，例如 `17.36`。
- 夏普比率：原始数值，例如 `0.51`。
- 最大回撤绝对值：百分比绝对值，例如 `20.15`。

优化结果会额外提示风险：

- `validation_score_negative`：验证区间 score 为负。
- `possible_overfit`：训练 score 明显高于验证 score。
- `too_few_trades`：验证区间交易次数低于最小交易次数。

## 训练/验证建议

- 不要只看全区间最优参数，优先看验证区间 score。
- 如果训练区间很好但验证区间变差，应视为可能过拟合。
- 如果交易次数过少，score 的可信度会下降。
- 分钟级和小时级数据有数据源限制，长区间优化建议优先使用日线。

## A 股交易规则

当前 Phase 2.0 UI 支持配置：

- 只做多。
- T+1。
- 100 股整数手。
- 涨跌停过滤。
- 成交量过滤。
- 滑点。
- 买入手续费。
- 卖出手续费。
- 印花税。
- 最小手续费。

## 测试

```bash
python -m pytest test/test_market_data_sources.py test/test_stock_search_cn.py test/test_market_insights_api.py test/test_index_template.py -q
python -m pytest test/test_optimization_runner.py test/test_optimize_api.py test/test_backtest_runner.py -q
python -m py_compile market_data.py stock_search.py main.py backtest_runner.py optimization_runner.py
```

## 开发约定

- 保持回测对象聚焦 A 股代码。
- 修改数据源、搜索或 UI 后运行聚焦测试。
- 完成并验证一个完整改动后及时提交。
