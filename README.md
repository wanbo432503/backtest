# 股票回测系统

Backtest 是一个面向 A 股代码的量化回测 Web 应用，基于 FastAPI、Backtesting.py 和免费公开数据源构建。当前产品口径只支持 A 股代码，不再支持美股、港股或加密货币作为回测对象。

## 核心功能

- A 股代码回测：支持 6 位代码、`SH/SZ/BJ` 前缀代码，以及 `.SH/.SZ/.BJ` 后缀代码。
- 策略回测：支持均线、RSI、布林带、动量等策略，通过策略注册表动态加载。
- 数据源选择：`auto` 默认优先使用 `mootdx`，失败后回退到 `yfinance` 的 A 股后缀代码。
- A 股搜索：支持通过 A 股代码和中文公司名称搜索，搜索结果统一回填 A 股代码。
- TradingAgents 智能分析：右侧面板提供 A 股研究分析、报告输出和 OpenAI-compatible 模型设置。
- Phase 2.0 策略优化：支持固定 1-3 只 A 股标的，按 `score` 对策略参数组合排序。
- Phase 3.1 组合选股回测：默认自动扫描 `60` / `00` 开头普通 A 股，按因子选 Top N（1-20），并按周、双周或月度调仓。
- Phase 3.1 因子优化：使用训练集 + 验证集，批量并行回测候选因子，默认按验证集“稳步抬升”目标排序，展示验证年化、波动、回撤、换手和过拟合风险。
- 风控与 A 股规则：支持止损、止盈、仓位比例、最大持仓周期、T+1、100 股整数手、涨跌停过滤、滑点、佣金、印花税等配置。
- Web 页面：左侧参数设置，中间统计和图表，右侧 TradingAgents 智能分析。

## 启动

```bash
./scripts/install_dependencies.sh
./scripts/start_server.sh
```

`scripts/install_dependencies.sh` 会把 backtest 和右侧 TradingAgents 智能分析所需依赖安装到当前激活的 Python 环境，并以 editable 方式挂载 `/Users/wanbo/knowledge/knowledge/repo/TradingAgents`。它不会创建单独的 TradingAgents Python 环境。

服务默认运行在 `http://localhost:8005`。`scripts/start_server.sh` 会先检查 8005 端口：如果已有进程在监听，会先停止它再启动新的 backtest 服务；如果没有进程监听，则直接启动。

如需临时改端口：

```bash
PORT=8010 ./scripts/start_server.sh
```

## 数据源

- `mootdx`：A 股 K 线主数据源，默认优先使用。
- `yfinance`：仅作为 A 股代码转换后的备用 K 线来源。
- TradingAgents：右侧智能分析由外部 TradingAgents 项目提供，代码来自 `/Users/wanbo/knowledge/knowledge/repo/TradingAgents`，配置来自该目录下的 `.env`，运行在当前 backtest Python 环境中。

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
    "strategy_name": "macd_volume_divergence_risk_control",
    "initial_cash": 10000,
    "commission": 0.002,
    "data_provider": "auto"
  }'
```

执行组合选股回测：

```bash
curl -X POST "http://localhost:8005/portfolio-backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "initial_cash": 100000,
    "data_provider": "auto",
    "universe": {
      "mode": "auto",
      "symbols": [],
      "max_scan_symbols": 200,
      "allowed_code_prefixes": ["60", "00"]
    },
    "selection": {
      "top_n": 5,
      "min_history_bars": 120
    },
    "rebalance": {
      "frequency": "monthly"
    }
  }'
```

创建组合因子优化任务：

```bash
curl -X POST "http://localhost:8005/portfolio-factor-optimization/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "base_request": {
      "start_date": "2024-01-01",
      "end_date": "2026-01-01",
      "initial_cash": 100000,
      "data_provider": "auto",
      "universe": {
        "mode": "auto",
        "symbols": [],
        "max_scan_symbols": 200,
        "allowed_code_prefixes": ["60", "00"]
      },
      "selection": {
        "top_n": 5,
        "min_history_bars": 120
      },
      "rebalance": {
        "frequency": "monthly"
      }
    },
    "split": {
      "method": "ratio",
      "train_ratio": 0.7
    },
    "search_space": {
      "momentum_lookback": [20, 40, 60, 90],
      "volatility_lookback": [10, 20],
      "liquidity_lookback": [10, 20],
      "momentum_weight": [0.35, 0.5],
      "volatility_weight": [-0.5, -0.25],
      "liquidity_weight": [0.15, 0.3],
      "trend_weight": [0, 0.1],
      "top_n": [3, 5, 10],
      "score_threshold": [null]
    },
    "max_trials": 64,
    "max_workers": 8,
    "executor_backend": "process",
    "objective": "validation_smooth_uptrend"
  }'
```

查询优化任务：

```bash
curl "http://localhost:8005/portfolio-factor-optimization/jobs/{job_id}"
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
      "symbols": ["SH603019"],
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

读取 TradingAgents 设置：

```bash
curl "http://localhost:8005/tradingagents/config"
```

保存 TradingAgents OpenAI-compatible 设置（API Key 留空表示不修改）：

```bash
curl -X PUT "http://localhost:8005/tradingagents/config" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai_compatible",
    "backend_url": "http://localhost:1234/v1",
    "api_key": null,
    "clear_api_key": false,
    "deep_model": "deep-model-name",
    "quick_model": "quick-model-name",
    "output_language": "Chinese",
    "max_debate_rounds": 1,
    "max_risk_rounds": 1,
    "checkpoint_enabled": false,
    "temperature": null,
    "openai_reasoning_effort": null
  }'
```

测试 TradingAgents 设置：

```bash
curl -X POST "http://localhost:8005/tradingagents/config/test"
```

发起 TradingAgents 分析：

```bash
curl -X POST "http://localhost:8005/tradingagents/analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SZ002241",
    "analysis_date": "2026-07-05",
    "analysts": ["market", "news", "fundamentals"],
    "max_debate_rounds": 1,
    "max_risk_rounds": 1
  }'
```

## Phase 3.1 组合选股与因子优化原型

Phase 3.1 面向“选股 + 组合交易 + 周期性调仓”的中低频原型。页面左侧的主入口是“组合选股回测”，默认从全部 `60` / `00` 普通 A 股中扫描；手动候选池只保留为高级诊断入口。

支持范围：

- 默认股票池为 `60` / `00` 开头普通 A 股。
- 自动扫描模式可从大股票池中选出 Top N，Top N 支持 1-20。
- 手动候选池仅用于小范围诊断，最多 4 只。
- 不支持科创板、创业板、北交所、基金、ETF、港股、美股或加密货币。
- 组合回测 MVP 仅使用日线数据。
- 因子优化使用训练集 + 验证集，不使用未来数据参与调仓日评分。
- 优化结果是参数候选和虚拟盘参考，不是自动实盘买卖指令。
- 免费数据源可用性会影响结果，数据缺口会进入 `data_warnings` 和 `risk_flags`。

推荐流程：

1. 使用自动扫描 `60/00` 股票池，设置 Top N、调仓周期、因子权重和组合风控。
2. 直接点击“开始组合回测”，查看当前参数下的净值、持仓、调仓、候选、成交和提示。
3. 打开“因子优化”，设置训练占比、最大试验数、并行 worker 和候选因子列表。
4. 点击“开始因子优化”，等待后台并行回测候选参数。
5. 优先比较验证年化、验证波动、下行波动、趋势R²、最大回撤、换手和风险标记。
6. 点击“应用参数”只会回填因子参数和 Top N，不会自动下单，也不会自动开始组合回测。
7. 再手动点击“开始组合回测”，用应用后的参数观察最新组合选择，可作为虚拟盘或人工复核参考。

## Phase 2.0 优化工作流

Phase 2.0 面向单只 A 股标的的集中交易参数优化场景，而不是股票池轮动或组合优化。优化时使用页面上方当前输入的 A 股代码，每次只搜索这一只股票的策略参数。

推荐流程：

1. 输入当前要优化的 A 股代码。
2. 选择策略，例如 RSI 风控、均线趋势风控或放量突破风控。
3. 设置风控参数和 A 股交易规则。
4. 启用参数优化，配置 Top N、最大组合数和最小交易次数。
5. 使用训练/验证切分查看 Top 结果。
6. 对候选参数点击“应用参数”或“回测该参数”，再看单次图表细节。

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
python -m pytest test/test_tradable_universe.py test/test_portfolio_models.py test/test_portfolio_data.py -q
python -m pytest test/test_factor_engine.py test/test_selection_engine.py test/test_portfolio_backtest_runner.py test/test_portfolio_api.py -q
python -m pytest test/test_portfolio_factor_optimization_models.py test/test_portfolio_factor_optimizer.py test/test_portfolio_factor_optimization_progress.py -q
python -m py_compile market_data.py stock_search.py main.py backtest_runner.py optimization_runner.py
```

## 开发约定

- 保持回测对象聚焦 A 股代码。
- 修改数据源、搜索或 UI 后运行聚焦测试。
- 完成并验证一个完整改动后及时提交。
