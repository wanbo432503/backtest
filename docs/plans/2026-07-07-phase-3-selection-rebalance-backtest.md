# Phase 3.0 Selection And Rebalance Backtest Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` or equivalent task-by-task execution discipline to implement this plan. Follow TDD, keep commits small, preserve A-share-only scope, and do not add unrelated portfolio complexity.

**Goal:** 将 backtest 从“单只股票 + 单策略参数回测”升级为“股票池选股 + 组合交易 + 周期性调仓”的中低频 A 股量化回测系统。

**Architecture:** 保留现有 FastAPI + Bootstrap 单页应用、`market_data.fetch_ohlcv(...)` 免费数据源入口、`analytics` 评分口径和 TradingAgents 智能分析面板。新增独立的组合回测薄层：股票池校验、因子计算、选股打分、调仓日历、组合成交模拟、结果汇总。不要把多标的组合逻辑塞进现有 `backtesting.py` 单标的 runner。

**Tech Stack:** FastAPI, Pydantic, pandas, numpy, mootdx/yfinance, Bootstrap, pytest, optional TradingAgents/LLM analysis snapshots.

---

## 1. 背景和当前事实

- 当前 `/backtest` 仍调用 `run_single_backtest(...)`，请求模型 `BacktestRequest` 只有一个 `symbol` 字段，见 `main.py:41` 和 `main.py:115`。
- 当前 `/optimize` 已有训练/验证、风险标记、按验证分排名的雏形，但 `OptimizationConfig.symbols` 被校验为只能包含一只股票，见 `optimization_models.py:56` 和 `optimization_models.py:80`。
- 当前 UI 左栏仍以单个 `#symbol` 输入框为中心，优化说明也写明“仅优化这一只股票”，见 `templates/index.html:536`、`templates/index.html:744` 和 `templates/index.html:1657`。
- 当前数据入口 `market_data.fetch_ohlcv(...)` 已限制为 A 股，并按 `auto -> mootdx -> yfinance` 方向工作，见 `market_data.py:311`。
- 当前 A 股规则 helper 已有涨跌停、T+1、整数手、手续费/印花税基础能力，见 `a_share_rules.py:18`、`a_share_rules.py:24`、`a_share_rules.py:38`。
- 当前右侧 TradingAgents 面板已能做单标的 AI 分析和配置管理，见 `templates/index.html:878` 和 `templates/index.html:1347`。

## 2. Phase 3.0 产品原则

1. **选股优先，交易规则简单。** 默认流程是“按因子打分选出 Top N，然后等权买入/调仓”，而不是为每只股票单独优化复杂策略。
2. **中低频、耐心持有。** 默认周期使用日线，调仓频率为每周、每两周或每月；分钟级和高频不进入 Phase 3.0。
3. **账户权限是硬约束。** 默认只允许代码以 `60` 或 `00` 开头的沪深股票；排除创业板、科创板、北交所、基金/ETF/LOF。
4. **免费数据源优先可复现。** MVP 支持用户维护的极小股票池，默认上限 4 只；不承诺全 A 5000 只每日稳定扫描。
5. **AI 做增强，不做不可复现的黑盒核心。** AI 可解释候选股票、生成研究摘要、离线写入 AI 因子快照；组合回测默认只依赖可复算的价格/成交量因子。
6. **风险胜过漂亮收益。** 所有结果必须展示验证分、回撤、换手、持仓集中度、交易次数、数据缺口和风险标记。

## 3. Phase 3.0 范围和非目标

### 3.1 本期目标

- 新增“选股组合回测”主流程：
  - 输入股票池。
  - 过滤不可交易标的。
  - 按技术/流动性/风险因子评分。
  - 每个调仓日选 Top N。
  - 等权或上限约束后的简单权重买入。
  - 周期性调仓并输出组合净值、持仓、交易、调仓日志。
- 前端从单个股票表单改造成组合回测工作台：
  - 股票池编辑器。
  - 选股规则配置。
  - 调仓和组合风控配置。
  - 组合回测结果面板。
  - 候选排名、调仓记录、成交记录。
- 保留单股回测能力作为“单股诊断/策略实验”次级入口，不再作为默认主流程。
- 后端新增 `/portfolio-backtest` API，不破坏旧 `/backtest` 和 `/optimize` 的基本兼容。
- 继续复用 TradingAgents 面板，但定位为“解释候选/单股研究/AI 快照”，不直接替代回测引擎。

### 3.2 非目标

- 不接实盘交易。
- 不做高频、盘口、做市、统计套利。
- 不做融资融券、做空、T+0、期货、基金、ETF、港股、美股、加密货币。
- 不做行业中性、指数增强、多因子回归、协方差矩阵优化、Barra 风险模型。
- 不做全市场 5000 只 A 股稳定扫描，除非后续有可靠证券列表和批量数据源。
- 不让 LLM 每次回测动态决定买卖，否则结果无法稳定复现。

## 4. 推荐方案

### 4.1 推荐选项: 极小股票池的中低频 Alpha 组合回测

**Approach:** 用户维护 2 到 4 只候选股票，系统只接受 `60` / `00` 开头股票，用可复算因子打分，每周或每月选 Top N 并等权调仓。

**Pros:**

- 符合你的耐心持有和不做高频的偏好。
- 免费数据源压力可控，调试也可控。
- 和现有 Phase 2.0 的 `score`、风险标记、A 股规则能自然衔接。
- 后续容易加入 AI 摘要、基本面快照、人工复核。

**Cons:**

- 不是完整机构级全市场选股。
- 选股质量受候选股票池质量限制。
- 免费数据源无法稳定覆盖停牌、复权、ST、指数成分变动等细节。

### 4.2 暂缓选项: 全 A 扫描式量化选股

**Approach:** 每天扫描全市场，按多因子排序选 Top N。

**Why not now:** 需要可靠证券列表、复权数据、财务因子、行业分类、停复牌、退市、ST 状态和批量抓取稳定性。以当前 mootdx/yfinance 免费数据源直接做，容易把数据缺口误当 Alpha。

### 4.3 暂缓选项: 复杂组合优化

**Approach:** 用风险模型、协方差矩阵、行业约束、权重优化决定每只股票仓位。

**Why not now:** 对个人中低频系统过重，且当前目标是先验证“选股 + 周期调仓”闭环。Phase 3.0 只做等权、最大单票权重和现金缓冲。

## 5. 后端数据模型

Create: `portfolio_models.py`

Core models:

```python
class UniverseConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SH603019", "SZ002241"])
    max_symbols: int = 4
    allowed_code_prefixes: tuple[str, ...] = ("60", "00")
    exclude_star: bool = True
    exclude_bj: bool = True
    exclude_funds: bool = True

class FactorConfig(BaseModel):
    momentum_lookback: int = 60
    volatility_lookback: int = 20
    liquidity_lookback: int = 20
    momentum_weight: float = 0.45
    volatility_weight: float = -0.25
    liquidity_weight: float = 0.20
    trend_weight: float = 0.10

class SelectionConfig(BaseModel):
    top_n: int = 2
    min_history_bars: int = 120
    min_avg_turnover_value: float | None = None
    score_threshold: float | None = None

class RebalanceConfig(BaseModel):
    frequency: Literal["weekly", "biweekly", "monthly"] = "monthly"
    weekday: int = 0
    monthday: int = 1
    lookahead_safe: bool = True

class PortfolioRiskConfig(BaseModel):
    max_position_pct: float = 0.50
    target_gross_exposure: float = 0.95
    cash_buffer_pct: float = 0.05
    stop_loss_pct: float | None = None
    max_drawdown_stop_pct: float | None = 30

class PortfolioBacktestRequest(BaseModel):
    start_date: str
    end_date: str
    initial_cash: float = 100000
    data_provider: str = "auto"
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    factors: FactorConfig = Field(default_factory=FactorConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    trading: AShareTradingConfig = Field(default_factory=AShareTradingConfig)
    risk: PortfolioRiskConfig = Field(default_factory=PortfolioRiskConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Response models should include:

- `summary`: total return, annual return, max drawdown, sharpe, win months, turnover, final equity.
- `equity_curve`: date, equity, cash, gross exposure, drawdown.
- `rebalance_events`: date, selected symbols, scores, sells, buys, skipped symbols and reasons.
- `positions`: date snapshots or compact per-rebalance holdings.
- `trades`: date, symbol, side, shares, price, amount, cost, reason.
- `candidate_rankings`: per rebalance date Top candidates with factor components.
- `data_warnings`: symbol-level missing data, fallback provider, insufficient history.
- `risk_flags`: high drawdown, high turnover, too concentrated, too few rebalances, data gaps.

## 6. Prototype implementation tasks

The implementation must produce a usable end-to-end prototype, not only backend modules. Tasks 0-10 are required for the Phase 3.0 prototype. Task 11 is optional AI polish and must not block the deterministic prototype.

### Task 0: Prototype contract and fixture data

**Purpose:** Define the API shape, stable demo inputs, and minimum visible prototype before changing behavior.

**Files:**

- Create: `test/fixtures/portfolio_ohlcv.py`
- Test: `test/test_phase3_prototype.py`

**Todo:**

- [x] Define one canonical demo request using `SH603019` and `SZ002241`, one-year daily data, monthly rebalance, `top_n=1`.
- [x] Define one invalid request with `SZ300750`, `SH688001`, and a fifth symbol to prove validation.
- [x] Create deterministic synthetic OHLCV fixture helpers for two to four symbols.
- [x] Fixtures must include at least 180 business-day rows, `Open/High/Low/Close/Volume`, and one symbol whose momentum becomes better mid-test.
- [x] Add a small helper to build DataFrames with optional limit-up/limit-down days.
- [x] Document the prototype completion rule: from the browser, user can run the demo pool and see summary, equity curve, holdings, rebalance records, candidate ranking, trades, warnings.

**Verification:**

- [x] Run `python -m pytest test/fixtures -q` if fixture tests are added, otherwise import the fixture helper from a tiny model test.

**Done when:**

- [x] Later tasks can reuse the same synthetic data without writing ad hoc frames in every test.

### Task 1: Tradable universe policy

**Purpose:** Enforce the real account boundary before any data fetch or backtest work.

**Files:**

- Create: `tradable_universe.py`
- Test: `test/test_tradable_universe.py`
- Modify: `stock_search.py`
- Modify: `main.py`

**Backend todo:**

- [x] Add `TradableUniversePolicy` with defaults: `max_symbols=4`, `allowed_code_prefixes=("60", "00")`, `exclude_funds=True`.
- [x] Add `TradableSymbolResult` or equivalent result object with `symbol`, `normalized_symbol`, `ok`, `reason`.
- [x] Implement `normalize_tradable_symbol(value)` accepting `600000`, `SH600000`, `600000.SH`, `000001`, `SZ000001`, `000001.SZ`.
- [x] Accept only normalized symbols whose six-digit code starts with `60` or `00`.
- [x] Reject `SZ300750`, `SZ301269`, `SH688001`, `SH689009`, `BJ430047`, `430047`, `830799`, `920001`.
- [x] Reject common fund/ETF prefixes: `SH510300`, `SH511880`, `SH512000`, `SZ159915`, `SZ160119`.
- [x] Reject duplicated symbols with a clear reason or dedupe them consistently before backtest. Prefer dedupe plus warning for UI friendliness.
- [x] Add `validate_universe(symbols, policy)` returning accepted symbols and rejected rows.
- [x] Update `stock_search.py` so portfolio-mode search results can expose whether a result is tradable; do not break existing `/search-stocks`.

**Frontend-facing contract todo:**

- [x] Rejection messages must be short and displayable: `only_60_00_prefix`, `too_many_symbols`, `fund_or_etf`, `unsupported_board`, `not_a_share`.
- [x] Accepted symbols should be returned in the same display form the UI uses, e.g. `SH603019`, `SZ002241`.

**Test todo:**

- [x] Test all allowed examples: `SH600000`, `600000`, `600000.SH`, `SH603019`, `SZ000001`, `000001`, `000001.SZ`, `SZ002241`.
- [x] Test all blocked board examples above.
- [x] Test more than 4 unique symbols fails.
- [x] Test duplicate handling.
- [x] Test portfolio search metadata for allowed and blocked symbols.

**Verification:**

- [x] Run `python -m pytest test/test_tradable_universe.py -q`.
- [x] Run `python -m pytest test/test_stock_search_cn.py -q`.

**Done when:**

- [x] The shared policy layer rejects symbols outside `60` / `00` and universes above 4 symbols before later Phase 3.0 endpoints call any data loader.

### Task 2: Portfolio models and response schema

**Purpose:** Create the stable request/response objects the backend, frontend, and tests will share.

**Files:**

- Create: `portfolio_models.py`
- Test: `test/test_portfolio_models.py`

**Backend todo:**

- [x] Implement `UniverseConfig`, `FactorConfig`, `SelectionConfig`, `RebalanceConfig`, `PortfolioRiskConfig`, `PortfolioBacktestRequest`.
- [x] Implement response row models or typed dict helpers for `PortfolioSummary`, `EquityPoint`, `PositionSnapshot`, `TradeRecord`, `CandidateScoreRow`, `RebalanceEvent`.
- [x] Default symbols to `["SH603019", "SZ002241"]`.
- [x] Default `max_symbols=4`, `top_n=2`, `max_position_pct=0.50`, `target_gross_exposure=0.95`.
- [x] Validate `top_n <= len(symbols)` and `len(symbols) <= 4`.
- [x] Validate `frequency` is `weekly`, `biweekly`, or `monthly`.
- [x] Validate factor lookbacks are positive integers and weights are finite numbers.
- [x] Validate `target_gross_exposure <= 1`, `cash_buffer_pct >= 0`, `max_position_pct <= 1`.
- [x] Make dates parseable as `YYYY-MM-DD` and require start < end.
- [x] Add `to_api_response()` helper on result dataclass if using dataclasses internally.

**Prototype response todo:**

- [x] Response must contain these top-level keys: `summary`, `equity_curve`, `positions`, `trades`, `rebalance_events`, `candidate_rankings`, `data_warnings`, `risk_flags`, `config`.
- [x] Every list key should return an empty list rather than `null`.
- [x] Numeric metrics should be raw numbers, not only formatted strings, so frontend can format them.

**Test todo:**

- [x] Test default request serializes.
- [x] Test invalid date order fails.
- [x] Test five symbols fail.
- [x] Test `top_n=3` with two symbols fails.
- [x] Test `SZ300750` fails model or policy validation before data loader.
- [x] Test response sample JSON includes all required keys.

**Verification:**

- [x] Run `python -m pytest test/test_portfolio_models.py -q`.

**Done when:**

- [x] The frontend can build a request and render a response without guessing missing fields.

### Task 3: Portfolio data loader

**Purpose:** Fetch small-pool OHLCV data through current free sources while preserving symbol-level warnings.

**Files:**

- Create: `portfolio_data.py`
- Test: `test/test_portfolio_data.py`
- Reuse: `market_data.py`

**Backend todo:**

- [x] Implement `load_portfolio_ohlcv(symbols, start_date, end_date, provider="auto", interval="1d")`.
- [x] Force Phase 3.0 portfolio backtests to `interval="1d"` for MVP.
- [x] Call `market_data.fetch_ohlcv(...)` per accepted symbol.
- [x] Call `prepare_ohlcv(...)` per symbol.
- [x] Return `PortfolioDataBundle(data_by_symbol, warnings)` or equivalent.
- [x] Include provider warnings and fallback warnings per symbol.
- [x] If one symbol fails, keep running with remaining symbols and record warning.
- [x] If all symbols fail, raise `ValueError` with a user-facing message.
- [x] Drop symbols with fewer rows than `selection.min_history_bars`, but report `insufficient_history`.
- [x] Align available dates by union for mark-to-market, and use previous known close only when explicitly documented.
- [x] Do not introduce persistent caching in prototype; use local variables only.

**Test todo:**

- [x] Monkeypatch `fetch_ohlcv` to return fixture frames for two symbols.
- [x] Test one-symbol fetch failure returns the other symbol plus warning.
- [x] Test all-symbol failure raises.
- [x] Test insufficient history warning.
- [x] Test data columns are prepared as `Open/High/Low/Close/Volume`.

**Verification:**

- [x] Run `python -m pytest test/test_portfolio_data.py -q`.

**Done when:**

- [x] The runner can receive clean per-symbol DataFrames and explicit data quality warnings.

### Task 4: Factor scoring engine

**Purpose:** Produce deterministic candidate rankings at each rebalance date without future data.

**Files:**

- Create: `factor_engine.py`
- Test: `test/test_factor_engine.py`

**Backend todo:**

- [ ] Implement `calculate_symbol_factors(data, as_of_date, config, lookahead_safe=True)`.
- [ ] Momentum: close return over `momentum_lookback`.
- [ ] Volatility: rolling std of daily returns over `volatility_lookback`; lower is better.
- [ ] Liquidity: rolling average volume or close*volume over `liquidity_lookback`; higher is better.
- [ ] Trend: close above moving average or moving-average slope.
- [ ] Implement `score_candidates(data_by_symbol, as_of_date, factor_config, selection_config)`.
- [ ] Normalize each factor cross-sectionally with deterministic behavior when all values are equal.
- [ ] Combine factors using configured weights.
- [ ] Return rows with `symbol`, `score`, `rank`, `factor_values`, `skip_reason`.
- [ ] When `lookahead_safe=True`, use only rows strictly before the rebalance trade date.
- [ ] Keep function pure: no fetching data, no portfolio state mutation.

**Test todo:**

- [ ] Test upward-trending symbol outranks downward symbol with positive momentum weight.
- [ ] Test high-volatility symbol is penalized.
- [ ] Test illiquid symbol is penalized or ranked lower when liquidity weight is positive.
- [ ] Test insufficient history produces `skip_reason="insufficient_history"`.
- [ ] Test no look-ahead by making trade-date close spike and confirming it does not affect same-day score.
- [ ] Test stable tie-breaking by symbol for equal scores.

**Verification:**

- [ ] Run `python -m pytest test/test_factor_engine.py -q`.

**Done when:**

- [ ] Every rebalance date can produce a transparent candidate ranking suitable for UI display.

### Task 5: Rebalance calendar and selection engine

**Purpose:** Turn daily data and candidate rankings into rebalance decisions.

**Files:**

- Create: `selection_engine.py`
- Test: `test/test_selection_engine.py`

**Backend todo:**

- [ ] Implement `build_trading_calendar(data_by_symbol)` using sorted union of available dates.
- [ ] Implement `build_rebalance_dates(calendar, start_date, end_date, config)`.
- [ ] Weekly: first available trading day on or after configured weekday.
- [ ] Biweekly: every other weekly rebalance date.
- [ ] Monthly: first available trading day on or after configured month day.
- [ ] Implement `select_top_candidates(candidate_rows, selection_config)`.
- [ ] Filter skipped candidates before selecting.
- [ ] Apply optional `score_threshold`.
- [ ] Return both selected rows and full ranking rows.
- [ ] Include a warning when fewer than `top_n` symbols are selectable.

**Test todo:**

- [ ] Test weekly dates on synthetic calendar.
- [ ] Test biweekly dates.
- [ ] Test monthly dates when month day falls on a non-trading day.
- [ ] Test `top_n=1` selects highest score.
- [ ] Test skipped symbols remain visible in full ranking.
- [ ] Test fewer selectable symbols than `top_n` produces warning but does not crash.

**Verification:**

- [ ] Run `python -m pytest test/test_selection_engine.py -q`.

**Done when:**

- [ ] The backtest runner can ask one function for rebalance dates and one function for selected symbols.

### Task 6: Portfolio execution simulator

**Purpose:** Produce realistic-enough prototype portfolio results: equity curve, positions, trades, rebalances, and risk flags.

**Files:**

- Create: `portfolio_backtest_runner.py`
- Test: `test/test_portfolio_backtest_runner.py`
- Reuse: `a_share_rules.py`
- Reuse: `analytics.py`
- Reuse: `portfolio_data.py`
- Reuse: `factor_engine.py`
- Reuse: `selection_engine.py`

**Backend todo:**

- [ ] Implement `run_portfolio_backtest(request: PortfolioBacktestRequest) -> PortfolioBacktestResult`.
- [ ] Validate universe with `validate_universe(...)` before data loading.
- [ ] Load portfolio OHLCV.
- [ ] Build trading calendar and rebalance dates.
- [ ] Initialize cash, empty positions, equity curve, trades, rebalance events, rankings, warnings.
- [ ] Each trading day, mark current positions to close price.
- [ ] On rebalance day, compute candidate scores using previous completed bars.
- [ ] Select Top N.
- [ ] Build target weights: equal weight capped by `max_position_pct`, total capped by `target_gross_exposure`.
- [ ] Sell removed or overweight positions before buys.
- [ ] Buy underweight target positions with available cash.
- [ ] Round shares to 100-lot using `round_lot_shares(...)`.
- [ ] Apply slippage using `apply_slippage(...)`.
- [ ] Apply commission/stamp tax using `calculate_trade_cost(...)`.
- [ ] Respect `can_buy(...)` and `can_sell(...)` for limit-up/down and T+1 where data allows.
- [ ] Record skipped trades with reason in rebalance event.
- [ ] Record final holdings snapshot even if no trade occurs.
- [ ] Compute summary: total return, annual return/CAGR, max drawdown, sharpe, final equity, turnover, number of rebalances, number of trades.
- [ ] Add risk flags: `too_few_rebalances`, `high_drawdown`, `high_turnover`, `data_gaps`, `underinvested`, `too_few_selected`.
- [ ] Return response-ready structures matching `portfolio_models.py`.

**Test todo:**

- [ ] Test two-symbol monthly backtest produces non-empty equity curve and at least one rebalance.
- [ ] Test selected symbol changes after momentum changes.
- [ ] Test cash decreases after buy including cost.
- [ ] Test shares are rounded to 100.
- [ ] Test max single position cap.
- [ ] Test sells happen before buys.
- [ ] Test limit-up buy skip.
- [ ] Test limit-down sell skip.
- [ ] Test T+1 same-day sell skip.
- [ ] Test final summary metrics are finite numbers.
- [ ] Test all response top-level lists exist.

**Verification:**

- [ ] Run `python -m pytest test/test_portfolio_backtest_runner.py -q`.

**Done when:**

- [ ] A pure Python call to `run_portfolio_backtest(...)` returns a complete prototype result without FastAPI or browser involvement.

### Task 7: Portfolio API endpoints

**Purpose:** Make the backend prototype callable from the browser with clear errors.

**Files:**

- Modify: `main.py`
- Test: `test/test_portfolio_api.py`

**Backend todo:**

- [ ] Import `PortfolioBacktestRequest`.
- [ ] Import `run_portfolio_backtest`.
- [ ] Add `POST /portfolio/validate-universe`.
- [ ] Add `POST /portfolio-backtest`.
- [ ] For validation errors, return HTTP 400 and Pydantic/API messages safe for UI display.
- [ ] For data-source failures, return HTTP 400 when user/actionable, HTTP 500 only for unexpected bugs.
- [ ] Do not remove `/backtest`, `/optimize`, `/strategies`, `/search-stocks`, or `/tradingagents/*`.
- [ ] Ensure endpoints execute CPU/data work in a threadpool if runtime is blocking.
- [ ] Include `data_warnings` and `risk_flags` in API response even when empty.

**Test todo:**

- [ ] Test `POST /portfolio/validate-universe` accepts `SH603019`, `SZ002241`.
- [ ] Test validation rejects `SZ300750`.
- [ ] Test validation rejects five symbols.
- [ ] Test `POST /portfolio-backtest` success using monkeypatched runner.
- [ ] Test runner `ValueError` maps to 400.
- [ ] Test unexpected exception maps to 500.
- [ ] Test old `/backtest` still returns expected shape with monkeypatch or existing fixture.

**Verification:**

- [ ] Run `python -m pytest test/test_portfolio_api.py -q`.
- [ ] Run `python -m pytest test/test_backtest_runner.py test/test_optimize_api.py -q`.

**Done when:**

- [ ] The browser can call `/portfolio-backtest` and receive a complete JSON result or a clear validation error.

### Task 8: Frontend portfolio workbench

**Purpose:** Replace the primary interaction with a usable portfolio-backtest form while keeping single-stock diagnostics available.

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`

**Frontend todo:**

- [ ] Rename the primary left card header to “组合选股回测”.
- [ ] Add `<form id="portfolioBacktestForm">` as the primary form.
- [ ] Add stock-pool textarea or chip editor with default `SH603019`, `SZ002241`.
- [ ] Add “加入股票池” action from search results.
- [ ] Add client-side validation for max 4 symbols.
- [ ] Add client-side validation for `60` / `00` prefix before API call.
- [ ] Add inline validation summary with accepted/rejected symbols.
- [ ] Add factor controls: momentum lookback, volatility lookback, liquidity lookback, factor weights.
- [ ] Add selection controls: Top N, min history bars, score threshold.
- [ ] Add rebalance controls: weekly, biweekly, monthly.
- [ ] Add portfolio controls: initial cash, max single position, target exposure, cash buffer.
- [ ] Reuse existing A-share rule controls where possible: T+1, limit filter, lot size, slippage, fees.
- [ ] Move existing single-stock strategy form into a collapsed “单股诊断” panel or secondary tab.
- [ ] Change primary submit button text to “开始组合回测”.
- [ ] Build `collectPortfolioRequest()` that matches `PortfolioBacktestRequest`.
- [ ] Submit to `/portfolio-backtest`.
- [ ] Keep existing TradingAgents panel and config behavior intact.

**Test todo:**

- [ ] Assert template contains `portfolioBacktestForm`.
- [ ] Assert template contains stock pool input.
- [ ] Assert template contains `collectPortfolioRequest`.
- [ ] Assert template contains `/portfolio-backtest`.
- [ ] Assert template still contains single-stock form or diagnostic section.
- [ ] Assert template still contains TradingAgents controls.

**Verification:**

- [ ] Run `python -m pytest test/test_index_template.py -q`.

**Done when:**

- [ ] The first screen lets a user configure and submit a portfolio backtest request without editing JSON manually.

### Task 9: Frontend result rendering

**Purpose:** Make the prototype useful after the API returns: the user can inspect performance, holdings, selections, trades, and warnings.

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`

**Frontend todo:**

- [ ] Add `renderPortfolioResult(result)`.
- [ ] Add summary metric strip: final equity, total return, annual return, max drawdown, sharpe, turnover.
- [ ] Add equity curve rendering. MVP may use inline SVG; no heavy chart dependency required.
- [ ] Add risk flag badges.
- [ ] Add data warning panel.
- [ ] Add Bootstrap tabs for:
  - [ ] 持仓
  - [ ] 调仓记录
  - [ ] 候选排名
  - [ ] 成交记录
  - [ ] 数据质量
- [ ] Render final/current holdings table from `positions`.
- [ ] Render `rebalance_events` with selected symbols and skipped trade reasons.
- [ ] Render `candidate_rankings` with score and factor components.
- [ ] Render `trades` with side, shares, price, amount, cost, reason.
- [ ] Add click action on candidate/holding symbol to sync `taSymbol`.
- [ ] Add empty states for no trades, no warnings, no rankings.
- [ ] Ensure long tables use horizontal scrolling and do not break the three-column layout.
- [ ] Update loading/error text to mention portfolio backtest.

**Test todo:**

- [ ] Assert template contains `renderPortfolioResult`.
- [ ] Assert template contains result tab IDs or labels.
- [ ] Assert template contains `candidate_rankings`, `rebalance_events`, and `trades` render references.
- [ ] Assert template contains TradingAgents sync action from result rows.

**Manual verification todo:**

- [ ] Start server.
- [ ] Run demo pool `SH603019`, `SZ002241`.
- [ ] Confirm visible summary metrics.
- [ ] Confirm equity curve appears.
- [ ] Confirm holdings/rebalance/candidate/trade tabs render.
- [ ] Confirm clicking a symbol syncs the right-side AI analysis symbol.

**Done when:**

- [ ] A browser user can run a portfolio backtest and understand what was bought, sold, selected, skipped, and why.

### Task 10: End-to-end prototype hardening

**Purpose:** Tie backend and frontend together into a prototype that starts cleanly, handles errors, and keeps old features alive.

**Files:**

- Modify: `README.md`
- Modify: `test/test_comprehensive.py` or create `test/test_phase3_prototype.py`
- Possibly modify: `scripts/start_server.sh`

**Todo:**

- [ ] Add one integration-style test that builds a real `PortfolioBacktestRequest` with monkeypatched data loader and checks API response shape.
- [ ] Add one template/API contract test: frontend expected top-level keys match backend response.
- [ ] Add README section “Phase 3.0 组合选股回测原型”.
- [ ] Document supported symbols: only `60` / `00`, max 4 symbols.
- [ ] Document default demo pool and smoke-test steps.
- [ ] Confirm start script still launches app.
- [ ] Run focused portfolio tests.
- [ ] Run old regression tests for single-stock backtest, optimization, search, TradingAgents config.
- [ ] Run full test suite.
- [ ] Manually start server and perform browser smoke test.

**Verification:**

- [ ] `python -m pytest test/test_tradable_universe.py test/test_portfolio_models.py test/test_portfolio_data.py -q`
- [ ] `python -m pytest test/test_factor_engine.py test/test_selection_engine.py test/test_portfolio_backtest_runner.py -q`
- [ ] `python -m pytest test/test_portfolio_api.py test/test_index_template.py -q`
- [ ] `python -m pytest test/test_backtest_runner.py test/test_optimize_api.py test/test_tradingagents_api.py -q`
- [ ] `python -m pytest -q`
- [ ] `./scripts/start_server.sh`

**Done when:**

- [ ] The Phase 3.0 prototype is usable from the browser, and backend tests prove the deterministic engine works without live data.

### Task 11: AI-assisted portfolio summary hook

**Purpose:** Add AI explanation only after the deterministic prototype is usable. This task must not block the prototype.

**Files:**

- Modify: `tradingagents_models.py`
- Modify: `tradingagents_adapter.py`
- Modify: `main.py`
- Modify: `templates/index.html`
- Test: `test/test_tradingagents_api.py`

**Todo:**

- [ ] Keep existing single-symbol TradingAgents analysis unchanged.
- [ ] Add portfolio summary request model accepting selected symbols, summary metrics, latest candidate rankings, and top risk flags.
- [ ] Add `POST /tradingagents/portfolio-summary` with deterministic prompt construction from backtest output.
- [ ] Never send or return API key values in this response.
- [ ] Add frontend “组合总结” button after a successful portfolio backtest.
- [ ] Store AI response as explanation text only; never feed it back into current backtest metrics.
- [ ] Add tests for request validation and secret masking.

**Verification:**

- [ ] Run `python -m pytest test/test_tradingagents_api.py -q`.

**Done when:**

- [ ] AI can explain a completed backtest, but disabling AI leaves the prototype fully functional.

## 7. Frontend interaction design

The first screen should be the portfolio workbench, not a landing page.

Left panel:

- 股票池:
  - multiline input and chips.
  - hard cap at 4 symbols.
  - only `60` / `00` prefixes accepted.
  - search by code/name.
  - validation summary: allowed, blocked, missing data.
- 选股 Alpha:
  - momentum, volatility, liquidity, trend weights.
  - lookback periods.
- 调仓:
  - weekly, biweekly, monthly.
  - Top N.
  - max single position.
  - target exposure/cash buffer.
- 交易约束:
  - T+1, lot size, limit filters, slippage, fees.
- Run button:
  - “开始组合回测”.

Center panel:

- Summary metrics.
- Equity curve and drawdown.
- Risk flags and data warnings.
- Tabs:
  - 持仓
  - 调仓记录
  - 候选排名
  - 成交记录
  - 数据质量

Right panel:

- Keep “智能分析”.
- Analysis tab should sync from:
  - selected portfolio candidate.
  - selected holding.
  - manually entered symbol.
- Add optional “组合总结” panel after portfolio run if Task 11 is implemented.

## 8. Data and factor limitations

Phase 3.0 must show these limitations in warnings rather than hiding them:

- `mootdx` and `yfinance` may disagree on adjusted prices, volume, missing days, and ticker availability.
- Free data may not include reliable ST status, suspension status, corporate actions, industry classification, or fundamental fields.
- When a symbol has insufficient bars, the ranking table should show `insufficient_history` instead of silently dropping it.
- When all data sources fail for a symbol, keep the symbol in `data_warnings`.
- If the user enters more than 4 symbols, the UI and API should reject the request before any data fetch.

## 9. Acceptance criteria

- A user can enter a 2 to 4 stock pool and run a monthly Top N portfolio backtest from the browser.
- Only `60` / `00` stock codes are accepted; `SZ300750`, `SH688001`, `BJ430047`, common fund/ETF prefixes, and non-A-share symbols are rejected before the expensive backtest runs.
- Portfolio backtest returns summary metrics, equity curve, rebalance events, positions, candidate rankings, trades, warnings, and risk flags.
- The selector never uses future data when computing a rebalance-day score.
- Trades respect integer lots, fees, slippage, T+1, and limit-up/down filters where data allows.
- Existing `/backtest`, `/optimize`, `/strategies`, `/search-stocks`, and `/tradingagents/*` tests still pass.
- UI default workflow is no longer “单股票 + 单策略参数”, but “股票池 + Alpha 选股 + 周期调仓”.
- AI output is clearly labeled as analysis/explanation and does not change deterministic backtest results unless imported as an explicit snapshot factor in a future phase.

## 10. Verification plan

Run focused tests after each task, then final verification:

```bash
python -m pytest test/test_tradable_universe.py test/test_portfolio_models.py test/test_portfolio_data.py -q
python -m pytest test/test_factor_engine.py test/test_selection_engine.py test/test_portfolio_backtest_runner.py -q
python -m pytest test/test_portfolio_api.py test/test_index_template.py test/test_phase3_prototype.py -q
python -m pytest test/test_backtest_runner.py test/test_optimize_api.py test/test_tradingagents_api.py -q
python -m pytest -q
```

Manual smoke test:

1. Start server with `./scripts/start_server.sh`.
2. Open the UI.
3. Use pool `SH603019`, `SZ002241`.
4. Run monthly rebalance, Top 1, one-year date range.
5. Confirm result has at least one rebalance event and no disallowed-symbol warning.
6. Add `SZ300750` and confirm UI/API rejects it because only `60` / `00` prefixes are tradable.
7. Add `SH688001` and confirm UI/API rejects it as 科创板.
8. Click a candidate and confirm TradingAgents symbol syncs to the right panel.

## 11. Suggested implementation order

0. Prototype contract and fixture data.
1. Eligibility policy.
2. Portfolio models.
3. Data loader.
4. Factor engine.
5. Selector and rebalance calendar.
6. Portfolio simulator.
7. API endpoint.
8. Frontend workbench.
9. Result rendering.
10. End-to-end prototype hardening.
11. Optional AI summary hook.

This order keeps the core deterministic engine testable before UI work begins.

## 12. Follow-up phases

Phase 3.1:

- Import/export stock pools.
- Add local JSON cache for OHLCV with refresh controls.
- Add benchmark comparison against沪深300/中证500 if reliable free index data is available.
- Add more factor families: gap/volume breakout, drawdown recovery, relative strength.

Phase 3.2:

- Optional AI factor snapshot import.
- Basic fundamental factors if a reliable free endpoint is chosen.
- Walk-forward portfolio validation.

Phase 4.0:

- Broker-independent signal export, not direct trading.
- Position recommendation report with manual execution checklist.
- More realistic corporate action and suspension handling if better data is available.
