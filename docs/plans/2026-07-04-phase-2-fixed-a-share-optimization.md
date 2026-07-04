# Phase 2.0 固定 A 股标的策略优化与风控回测实施计划

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` or equivalent task-by-task execution discipline to implement this plan. Follow TDD, make small commits, and do not bundle unrelated refactors.

**Goal:** 将当前单次回测工具升级为“固定 2-3 只 A 股标的的策略参数优化器 + 风控回测器”，用统一 `score` 评估不同策略与参数组合，帮助用户判断“该怎么交易”。

**Architecture:** 保留 `backtesting.py` 作为 Phase 2.0 执行引擎，但新增 Strategy Config、Risk Engine、A-share Rule Engine、Optimization Runner、Result Reporter 五个薄层。业务逻辑不要写死在 `backtesting.py Strategy` 内，后续如需迁移到 RQAlpha/Backtrader/vectorbt，可以替换 Backtest Adapter。

**Tech Stack:** FastAPI, Backtesting.py, pandas, numpy, mootdx/yfinance A 股数据源, Bootstrap UI, pytest.

---

## 1. 范围和非目标

### 1.1 本期目标

- 固定股票池：最多 2-3 只 A 股，默认支持中科曙光 `SH603019` 和歌尔股份 `SZ002241`。
- 策略优化目标：最大化自定义 `score`。
- 支持训练/验证切分，避免只看全区间过拟合结果。
- 支持风控配置：止损、止盈、仓位比例、最大持仓周期、冷却期、最大回撤停止开仓。
- 支持 A 股交易规则：只做多、T+1、100 股整数手、涨跌停过滤、成交量过滤、滑点、佣金、印花税、最小手续费。
- UI 能配置股票池、策略、风控、A 股规则、参数优化，并展示 Top N 优化结果。

### 1.2 非目标

- 不做多股票组合轮动。
- 不做行业中性、组合权重优化、指数增强。
- 不接实盘交易。
- 不迁移到大型量化框架。
- 不追求毫秒级高频撮合。

---

## 2. 评分公式

Phase 2.0 统一使用以下简单 score：

```text
score = 年化收益率 * 0.4 + 夏普比率 * 0.3 - 最大回撤绝对值 * 0.3
```

字段口径：

- 年化收益率：百分比数值，例如 `17.36`。
- 夏普比率：原始数值，例如 `0.51`。
- 最大回撤绝对值：百分比绝对值，例如 `20.15`。

硬性过滤建议：

- 交易次数 `< min_trades` 的结果不参与排名，默认 `min_trades = 5`。
- 最大回撤超过阈值的结果标记为高风险，默认阈值 `30%`。
- 验证区间 score 为负时，不推荐，即使训练区间 score 很高。

---

## 3. 当前代码入口

- `main.py`
  - `BacktestRequest`: 当前回测请求模型。
  - `/backtest`: 当前单次回测 API。
  - `/strategies`: 策略列表 API。
- `market_data.py`
  - `fetch_ohlcv(...)`: 当前 A 股 K 线数据统一入口。
  - `normalize_symbol(...)`: 当前 A 股代码规范化入口。
- `templates/index.html`
  - 当前 Web UI 主页面。
  - 左侧回测配置、中间统计/图表、右侧标的信息。
- `market_insights.py`
  - 当前研报、资金、龙虎榜、公告数据面板。
- `strategies/`
  - 当前动态策略目录，已有突破动量、智能对冲等策略。
- `test/`
  - 当前测试目录。

---

## 4. Phase 2.0 数据模型设计

### 4.1 StrategyParamConfig

用途：描述策略参数及优化范围。

字段：

- `strategy_name: str`
- `fixed_params: dict[str, int | float | str | bool]`
- `search_space: dict[str, list[int | float | str | bool]]`

示例：

```json
{
  "strategy_name": "rsi_risk_control",
  "fixed_params": {
    "trend_ma": 60
  },
  "search_space": {
    "rsi_period": [6, 14, 21],
    "rsi_buy": [25, 30, 35],
    "rsi_sell": [60, 70, 80],
    "stop_loss_pct": [3, 5, 8],
    "take_profit_pct": [8, 12, 20]
  }
}
```

### 4.2 RiskConfig

字段：

- `enabled: bool`
- `position_pct: float`
- `stop_loss_pct: float | None`
- `take_profit_pct: float | None`
- `max_holding_bars: int | None`
- `cooldown_bars: int`
- `max_account_drawdown_pct: float | None`
- `atr_stop_enabled: bool`
- `atr_period: int`
- `atr_multiplier: float`

默认：

```json
{
  "enabled": true,
  "position_pct": 0.95,
  "stop_loss_pct": 5,
  "take_profit_pct": 12,
  "max_holding_bars": 120,
  "cooldown_bars": 3,
  "max_account_drawdown_pct": 30,
  "atr_stop_enabled": false,
  "atr_period": 14,
  "atr_multiplier": 2.0
}
```

### 4.3 AShareTradingConfig

字段：

- `long_only: bool`
- `t_plus_one: bool`
- `lot_size: int`
- `limit_up_down_filter: bool`
- `volume_filter: bool`
- `min_volume: float`
- `slippage_pct: float`
- `buy_commission_pct: float`
- `sell_commission_pct: float`
- `stamp_tax_pct: float`
- `min_commission: float`

默认：

```json
{
  "long_only": true,
  "t_plus_one": true,
  "lot_size": 100,
  "limit_up_down_filter": true,
  "volume_filter": true,
  "min_volume": 1,
  "slippage_pct": 0.05,
  "buy_commission_pct": 0.03,
  "sell_commission_pct": 0.03,
  "stamp_tax_pct": 0.05,
  "min_commission": 5
}
```

### 4.4 OptimizationConfig

字段：

- `enabled: bool`
- `symbols: list[str]`
- `strategies: list[StrategyParamConfig]`
- `objective: str`
- `top_n: int`
- `max_combinations: int`
- `min_trades: int`
- `train_start_date: str`
- `train_end_date: str`
- `validate_start_date: str`
- `validate_end_date: str`
- `interval: str`
- `data_provider: str`

默认：

```json
{
  "enabled": false,
  "symbols": ["SH603019", "SZ002241"],
  "objective": "score",
  "top_n": 10,
  "max_combinations": 300,
  "min_trades": 5
}
```

---

## 5. 架构任务拆分

## Task 1: 新增 score 和统计工具

**Files:**

- Create: `analytics.py`
- Test: `test/test_analytics.py`

**Goal:** 提供统一 score 计算和统计字段解析，后续单次回测、优化结果都使用同一口径。

**Todo:**

- [ ] 创建 `analytics.py`。
- [ ] 实现 `parse_percent(value: str | float) -> float`。
- [ ] 实现 `calculate_score(cagr_pct, sharpe, max_drawdown_pct) -> float`。
- [ ] 实现 `extract_core_metrics(stats: pd.Series | dict) -> dict`。
- [ ] 最大回撤统一转绝对值。
- [ ] 对 `nan` 夏普做容错，默认按 `0` 处理。
- [ ] 增加 `score` 字段。
- [ ] 写测试：正常 score。
- [ ] 写测试：最大回撤为负数时取绝对值。
- [ ] 写测试：夏普为空/NaN 时不报错。
- [ ] 运行 `python -m pytest test/test_analytics.py -q`。
- [ ] Commit: `feat: add score analytics helpers`。

**Acceptance Criteria:**

- 同一组收益、夏普、回撤输入总是产出稳定 score。
- 所有回测结果可以用同一 helper 计算 score。

---

## Task 2: 新增配置模型

**Files:**

- Modify: `main.py`
- Create: `optimization_models.py`
- Test: `test/test_optimization_models.py`

**Goal:** 将风控、A 股规则、参数优化配置结构化，避免继续在 API 里堆散字段。

**Todo:**

- [ ] 创建 `optimization_models.py`。
- [ ] 定义 `RiskConfig` Pydantic model。
- [ ] 定义 `AShareTradingConfig` Pydantic model。
- [ ] 定义 `StrategyParamConfig` Pydantic model。
- [ ] 定义 `OptimizationConfig` Pydantic model。
- [ ] 定义 `OptimizationRequest` Pydantic model。
- [ ] 在 `BacktestRequest` 增加可选 `risk_config`。
- [ ] 在 `BacktestRequest` 增加可选 `a_share_config`。
- [ ] 给所有 config 提供安全默认值。
- [ ] 校验 `position_pct` 必须在 `(0, 1]`。
- [ ] 校验 `lot_size` 默认 100 且必须大于 0。
- [ ] 校验 `max_combinations` 不超过 1000。
- [ ] 写模型默认值测试。
- [ ] 写非法参数测试。
- [ ] 运行 `python -m pytest test/test_optimization_models.py -q`。
- [ ] Commit: `feat: add phase 2 config models`。

**Acceptance Criteria:**

- API 可以接收新配置。
- 旧请求不传配置时仍能按默认配置运行。

---

## Task 3: 策略参数注册和 UI 元数据

**Files:**

- Modify: `main.py`
- Create: `strategy_metadata.py`
- Test: `test/test_strategy_metadata.py`

**Goal:** 让 UI 知道每个策略有哪些可配置参数和可优化参数。

**Todo:**

- [ ] 创建 `strategy_metadata.py`。
- [ ] 定义 `StrategyParamMeta`。
- [ ] 定义 `StrategyMeta`。
- [ ] 为现有 `sma_cross` 添加参数元数据：`n1`, `n2`。
- [ ] 为现有 `rsi` 添加参数元数据：`rsi_period`, `rsi_lower`, `rsi_upper`。
- [ ] 为新增策略预留 `rsi_risk_control` 元数据。
- [ ] `/strategies` API 返回参数元数据。
- [ ] UI 切换策略时能读取参数元数据。
- [ ] 写测试：`/strategies` 包含参数定义。
- [ ] 写测试：每个参数有 label、type、default、search_values。
- [ ] Commit: `feat: expose strategy parameter metadata`。

**Acceptance Criteria:**

- UI 不需要写死策略参数。
- 新增策略时只需要补 metadata。

---

## Task 4: 实现 RSI 风控版策略

**Files:**

- Create: `strategies/rsi_risk_control.py`
- Modify: `strategy_metadata.py`
- Test: `test/test_rsi_risk_control_strategy.py`

**Goal:** 提供第一个 Phase 2.0 标准策略：RSI + 趋势过滤 + 风控。

**Strategy Logic:**

- 买入条件：
  - 无持仓。
  - Close 在趋势均线上方，默认 `trend_ma = 60`。
  - RSI 从低位上穿 `rsi_buy`。
  - 未处于冷却期。
- 卖出条件：
  - RSI 上穿 `rsi_sell`。
  - 或固定止损触发。
  - 或固定止盈触发。
  - 或最大持仓周期触发。
  - 或趋势跌破趋势均线。

**Todo:**

- [ ] 写失败测试：RSI 上穿买入阈值时买入。
- [ ] 写失败测试：跌破止损时平仓。
- [ ] 写失败测试：达到止盈时平仓。
- [ ] 写失败测试：最大持仓周期强制平仓。
- [ ] 实现 `RSIRiskControlStrategy`。
- [ ] 支持参数：`rsi_period`, `rsi_buy`, `rsi_sell`, `trend_ma`, `stop_loss_pct`, `take_profit_pct`, `max_holding_bars`, `position_pct`, `cooldown_bars`。
- [ ] 禁止重复开仓。
- [ ] 冷却期内不允许再开仓。
- [ ] 在 `strategy_metadata.py` 注册。
- [ ] 在 `strategies.json` 中加入显示名称。
- [ ] 运行策略测试。
- [ ] 运行一次 API 回测 smoke test。
- [ ] Commit: `feat: add RSI risk-control strategy`。

**Acceptance Criteria:**

- 新策略出现在 UI 策略下拉框。
- 可通过 API 回测。
- 风控条件能改变交易行为。

---

## Task 5: 实现均线趋势风控版策略

**Files:**

- Create: `strategies/ma_trend_risk_control.py`
- Modify: `strategy_metadata.py`
- Test: `test/test_ma_trend_risk_control_strategy.py`

**Goal:** 提供趋势型策略，适合中科曙光、歌尔股份这种波动较大的趋势股。

**Strategy Logic:**

- 买入条件：
  - 短均线上穿长均线。
  - Close 在长期均线上方。
  - 最近 N 根 K 线涨幅为正。
- 卖出条件：
  - 短均线下穿长均线。
  - 固定止损/止盈。
  - 最大持仓周期。

**Todo:**

- [ ] 写失败测试：金叉买入。
- [ ] 写失败测试：死叉卖出。
- [ ] 写失败测试：趋势过滤阻止买入。
- [ ] 实现策略。
- [ ] 参数：`fast_ma`, `slow_ma`, `trend_ma`, `momentum_lookback`, `stop_loss_pct`, `take_profit_pct`, `position_pct`。
- [ ] 注册元数据。
- [ ] 运行测试。
- [ ] Commit: `feat: add MA trend risk-control strategy`。

**Acceptance Criteria:**

- 策略支持参数优化。
- 策略有基本趋势过滤和风控。

---

## Task 6: 实现放量突破风控版策略

**Files:**

- Create: `strategies/volume_breakout_risk_control.py`
- Modify: `strategy_metadata.py`
- Test: `test/test_volume_breakout_risk_control_strategy.py`

**Goal:** 提供适配 A 股强势股的突破策略。

**Strategy Logic:**

- 买入条件：
  - Close 突破过去 N 根 K 线高点。
  - 成交量大于过去 M 根均量的倍数。
  - 非涨停不可成交场景。
- 卖出条件：
  - 跌破突破线。
  - 止损/止盈。
  - 最大持仓周期。

**Todo:**

- [ ] 写失败测试：突破且放量买入。
- [ ] 写失败测试：无量突破不买入。
- [ ] 写失败测试：涨停过滤阻止买入。
- [ ] 实现策略。
- [ ] 参数：`breakout_lookback`, `volume_lookback`, `volume_multiplier`, `stop_loss_pct`, `take_profit_pct`, `position_pct`。
- [ ] 注册元数据。
- [ ] 运行测试。
- [ ] Commit: `feat: add volume breakout strategy`。

**Acceptance Criteria:**

- 策略能利用成交量。
- 可用于参数优化。

---

## Task 7: A 股规则引擎

**Files:**

- Create: `a_share_rules.py`
- Test: `test/test_a_share_rules.py`

**Goal:** 把 A 股交易规则从策略中抽出来，形成可测、可复用的规则层。

**Functions:**

- `is_limit_up(current_close, previous_close, threshold=0.1) -> bool`
- `is_limit_down(current_close, previous_close, threshold=0.1) -> bool`
- `round_lot_shares(raw_shares, lot_size=100) -> int`
- `can_buy(row, previous_close, config) -> tuple[bool, str | None]`
- `can_sell(row, previous_close, holding_bars, config) -> tuple[bool, str | None]`
- `apply_slippage(price, side, slippage_pct) -> float`
- `calculate_trade_cost(amount, side, config) -> float`

**Todo:**

- [ ] 写测试：涨停检测。
- [ ] 写测试：跌停检测。
- [ ] 写测试：100 股整数手。
- [ ] 写测试：买入滑点增加成本。
- [ ] 写测试：卖出滑点降低成交价。
- [ ] 写测试：买入佣金。
- [ ] 写测试：卖出佣金 + 印花税。
- [ ] 写测试：最小手续费。
- [ ] 实现 helper。
- [ ] Commit: `feat: add A-share rule helpers`。

**Acceptance Criteria:**

- A 股交易规则不依赖具体策略。
- 所有核心规则都有单元测试。

---

## Task 8: 回测适配层

**Files:**

- Create: `backtest_runner.py`
- Modify: `main.py`
- Test: `test/test_backtest_runner.py`

**Goal:** 将 `/backtest` 中的数据获取、回测执行、统计解析、score 计算拆出来，避免主路由继续膨胀。

**Responsibilities:**

- 接收 symbol、date、interval、strategy、config。
- 调用 `fetch_ohlcv`。
- 准备数据。
- 创建 `Backtest`。
- 执行 `bt.run()`。
- 生成图表 HTML。
- 计算 metrics 和 score。
- 返回统一 `BacktestResult`。

**Todo:**

- [ ] 创建 `BacktestResult` dataclass 或 Pydantic model。
- [ ] 创建 `run_single_backtest(...)`。
- [ ] 将 `main.py /backtest` 逻辑迁移到 runner。
- [ ] 保持 API 响应兼容现有 UI。
- [ ] 新增 `score` 到 stats。
- [ ] 测试 runner 返回 score。
- [ ] 测试错误数据抛出可读错误。
- [ ] 测试旧 `/backtest` API 仍返回 plot/stats/provider。
- [ ] Commit: `refactor: extract backtest runner`。

**Acceptance Criteria:**

- `/backtest` 变薄。
- 单次回测和参数优化复用同一 runner。

---

## Task 9: 参数优化 Runner

**Files:**

- Create: `optimization_runner.py`
- Modify: `main.py`
- Test: `test/test_optimization_runner.py`

**Goal:** 为固定 2-3 只股票执行策略参数搜索，按 score 排名。

**Core Functions:**

- `expand_search_space(search_space, max_combinations) -> list[dict]`
- `score_backtest_result(result) -> float`
- `run_optimization(request) -> OptimizationResult`
- `run_train_validate(symbol, strategy, params, config) -> dict`

**Todo:**

- [ ] 写测试：参数组合展开。
- [ ] 写测试：超过 `max_combinations` 时截断或报错。
- [ ] 写测试：交易次数低于 `min_trades` 时标记 `filtered`。
- [ ] 写测试：按验证 score 排序。
- [ ] 写测试：训练 score 高但验证 score 负时不推荐。
- [ ] 实现参数展开。
- [ ] 对每个 symbol、strategy、params 跑训练回测。
- [ ] 对 Top 候选跑验证回测。
- [ ] 生成结果表：symbol、strategy、params、train_metrics、validate_metrics、score、risk_flags。
- [ ] 支持 `top_n`。
- [ ] 支持进度日志。
- [ ] Commit: `feat: add parameter optimization runner`。

**Acceptance Criteria:**

- 能对 `SH603019` 和 `SZ002241` 跑 Top 10 优化结果。
- 单次请求组合数可控，不会卡死服务。

---

## Task 10: 新增 `/optimize` API

**Files:**

- Modify: `main.py`
- Test: `test/test_optimize_api.py`

**Goal:** UI 能触发优化任务并获取排名结果。

**Endpoint:**

```text
POST /optimize
```

**Request:** `OptimizationRequest`

**Response:**

```json
{
  "objective": "score",
  "symbols": ["SH603019", "SZ002241"],
  "top_results": [
    {
      "rank": 1,
      "symbol": "SH603019",
      "strategy_name": "rsi_risk_control",
      "params": {},
      "train_score": 4.12,
      "validate_score": 2.31,
      "validate_stats": {},
      "risk_flags": []
    }
  ],
  "warnings": []
}
```

**Todo:**

- [ ] 写失败测试：合法请求返回 200。
- [ ] 写失败测试：非 A 股 symbol 返回 400。
- [ ] 写失败测试：max_combinations 过大返回 400。
- [ ] 接入 `run_optimization`。
- [ ] 错误信息保持中文可读。
- [ ] Commit: `feat: expose optimize API`。

**Acceptance Criteria:**

- `/optimize` 可从 UI 调用。
- 错误不会返回 Python traceback。

---

## Task 11: UI - 股票池与优化设置

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`

**Goal:** UI 支持固定 2-3 只股票优化，而不是只能当前输入框单股回测。

**UI Changes:**

- 新增“股票池”区域：
  - checkbox: 中科曙光 `SH603019`
  - checkbox: 歌尔股份 `SZ002241`
  - 可选第三只输入框，必须是 A 股代码。
- 新增“优化设置”折叠区：
  - 启用参数优化 toggle。
  - 优化目标固定 `score`。
  - Top N。
  - 最大组合数。
  - 最小交易次数。
  - 训练/验证日期。

**Todo:**

- [ ] 写模板测试：页面含“股票池”。
- [ ] 写模板测试：页面含中科曙光、歌尔股份 checkbox。
- [ ] 写模板测试：页面含“启用参数优化”。
- [ ] 写模板测试：页面含 `score`。
- [ ] 实现 UI 区域。
- [ ] JS 收集 OptimizationRequest。
- [ ] 参数优化启用时显示“开始优化”按钮。
- [ ] 参数优化禁用时保留现有“开始回测”。
- [ ] Commit: `feat: add optimization controls to UI`。

**Acceptance Criteria:**

- 用户无需写 JSON 即可配置优化。
- 不破坏现有单股回测路径。

---

## Task 12: UI - 策略参数与风控设置

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`

**Goal:** 页面可配置策略参数、风控参数和 A 股规则。

**UI Controls:**

- 策略参数：
  - 根据 `/strategies` 元数据动态渲染。
  - 支持固定值和优化范围。
- 风控设置：
  - 止损比例。
  - 止盈比例。
  - 仓位比例。
  - 最大持仓周期。
  - 冷却期。
  - 最大回撤停止开仓。
- A 股规则：
  - T+1。
  - 100 股整数手。
  - 涨跌停过滤。
  - 成交量过滤。
  - 滑点。
  - 买入手续费。
  - 卖出手续费。
  - 印花税。
  - 最小手续费。

**Todo:**

- [ ] 写模板测试：页面含“风控设置”。
- [ ] 写模板测试：页面含“A 股交易规则”。
- [ ] 写模板测试：页面含“T+1”。
- [ ] 写模板测试：页面含“涨跌停过滤”。
- [ ] 实现折叠 UI。
- [ ] JS 将配置写入 `/backtest` 和 `/optimize` 请求。
- [ ] 默认值与后端 model 保持一致。
- [ ] Commit: `feat: add risk and A-share rule controls`。

**Acceptance Criteria:**

- 用户可在页面修改关键风控参数。
- 默认配置即可直接运行。

---

## Task 13: UI - 优化结果表

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`

**Goal:** 展示 Top N 参数组合，并允许一键应用最佳参数。

**Columns:**

- Rank
- 股票
- 策略
- 验证 score
- 训练 score
- 年化收益
- 最大回撤
- 夏普
- 交易次数
- 参数摘要
- 操作：应用并回测

**Todo:**

- [ ] 写模板测试：页面含“优化结果”。
- [ ] 写 JS 渲染 `top_results`。
- [ ] 高亮验证 score 最高的结果。
- [ ] 风险 flags 用黄色 badge。
- [ ] “应用并回测”将参数回填到左侧表单。
- [ ] 点击后触发单次 `/backtest`。
- [ ] Commit: `feat: render optimization result table`。

**Acceptance Criteria:**

- 用户能看到参数排名。
- 用户能把最佳参数用于单次图表回测。

---

## Task 14: 训练/验证切分与过拟合提示

**Files:**

- Modify: `optimization_runner.py`
- Modify: `templates/index.html`
- Test: `test/test_optimization_runner.py`

**Goal:** 防止“训练区间好看、验证区间失效”的参数被误当成最佳。

**Rules:**

- 默认按用户指定日期切分。
- 如果验证 score < 0，标记 `validation_score_negative`。
- 如果训练 score - 验证 score > 阈值，标记 `possible_overfit`。
- 如果交易次数过少，标记 `too_few_trades`。

**Todo:**

- [ ] 实现 risk flags。
- [ ] 测试验证 score 为负。
- [ ] 测试训练/验证差异过大。
- [ ] 测试交易次数不足。
- [ ] UI 显示风险提示。
- [ ] Commit: `feat: flag overfitting in optimization results`。

**Acceptance Criteria:**

- Top 结果不是只按训练 score 推荐。
- UI 明确提示可能过拟合。

---

## Task 15: 文档与项目约定更新

**Files:**

- Modify: `README.md`
- Modify: `AGENTS.md`
- Test: none

**Goal:** 记录 Phase 2.0 的新能力、score 口径和后续开发约束。

**Todo:**

- [ ] README 增加 Phase 2.0 使用说明。
- [ ] README 增加 score 公式。
- [ ] README 增加训练/验证建议。
- [ ] README 增加 A 股交易规则说明。
- [ ] AGENTS.md 增加“策略优化优先考虑固定少数 A 股标的”。
- [ ] AGENTS.md 增加“不要引入多股票组合复杂度，除非用户明确要求”。
- [ ] Commit: `docs: document phase 2 optimization workflow`。

**Acceptance Criteria:**

- 新人能通过 README 理解 Phase 2.0 如何使用。
- 后续 agent 不会误把方向带回多股票组合。

---

## 6. UI 最终信息架构

左侧：

- A 股代码输入。
- 快速选择：中科曙光、歌尔股份。
- 股票池。
- 回测策略。
- 策略参数。
- 风控设置。
- A 股交易规则。
- 数据频率/数据源。
- 开始回测。
- 开始优化。

中间：

- 回测统计。
- 交易规则摘要。
- 回测图表。
- 优化结果 Top N。
- 训练/验证对比。

右侧：

- 标的信息。
- 研报。
- 资金。
- 龙虎榜。
- 公告。
- warnings。

---

## 7. 验收标准

### 功能验收

- [ ] 用户可以只选择中科曙光、歌尔股份运行优化。
- [ ] 用户可以配置至少 2 个策略参与优化。
- [ ] 用户可以配置止损、止盈、仓位、最大持仓周期。
- [ ] 用户可以配置 T+1、涨跌停过滤、滑点、手续费。
- [ ] 优化结果按验证 score 排序。
- [ ] UI 展示训练 score 和验证 score。
- [ ] UI 可以一键应用最佳参数并生成回测图。

### 数据和交易规则验收

- [ ] 非 A 股代码仍被拒绝。
- [ ] 涨停过滤有测试覆盖。
- [ ] 跌停过滤有测试覆盖。
- [ ] 100 股整数手有测试覆盖。
- [ ] T+1 有测试覆盖。
- [ ] 交易成本拆分有测试覆盖。

### 测试验收

- [ ] `python -m pytest test/test_analytics.py -q`
- [ ] `python -m pytest test/test_optimization_models.py -q`
- [ ] `python -m pytest test/test_a_share_rules.py -q`
- [ ] `python -m pytest test/test_optimization_runner.py -q`
- [ ] `python -m pytest test/test_optimize_api.py -q`
- [ ] `python -m pytest test/test_index_template.py -q`
- [ ] `python -m py_compile main.py market_data.py market_insights.py stock_search.py`

---

## 8. 风险与缓解

### 风险 1: 参数优化过拟合

缓解：

- 强制训练/验证切分。
- 默认按验证 score 排名。
- 增加过拟合 flags。
- 最少交易次数过滤。

### 风险 2: 优化组合过多导致请求很慢

缓解：

- `max_combinations` 默认 300。
- API 限制最大 1000。
- UI 显示组合数预估。
- Phase 2.0 先同步执行，后续再考虑后台任务。

### 风险 3: A 股规则和 backtesting.py 撮合模型冲突

缓解：

- A 股规则先做 pre-trade filter 和成本修正。
- 不追求完整交易所队列模型。
- 后续如确实需要再抽换 Backtest Adapter。

### 风险 4: UI 复杂度过高

缓解：

- 使用折叠区。
- 默认值可直接运行。
- 高级配置默认收起。

---

## 9. 推荐执行顺序

1. Task 1: score helpers。
2. Task 2: config models。
3. Task 8: backtest runner 抽取。
4. Task 4: RSI 风控版策略。
5. Task 7: A 股规则 helper。
6. Task 9: optimization runner。
7. Task 10: `/optimize` API。
8. Task 11-13: UI。
9. Task 14: 训练/验证和过拟合提示。
10. Task 15: 文档更新。

原因：

- 先统一 score 和配置模型，避免后续重复返工。
- 先抽 runner，优化器才能复用单次回测逻辑。
- 先做一个策略打通闭环，再扩展其他策略。
- UI 放在 API 稳定后做，减少联调返工。

---

## 10. 每日/每阶段提交建议

- 每个 Task 完成并验证后单独 commit。
- commit message 使用：
  - `feat: ...`
  - `fix: ...`
  - `refactor: ...`
  - `test: ...`
  - `docs: ...`
- 禁止把 UI、策略、优化 runner、文档混在一个大 commit。

