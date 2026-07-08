# Phase 3.2 Selection Strategy Library Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a portfolio selection strategy library so users choose an understandable stock-selection template first, then optimize the template's factor windows, weights, filters, and Top N through the existing Phase 3.1 train/validation rolling-rebalance optimizer.

**Architecture:** Keep Phase 3.1's portfolio backtest and factor optimization runner as the execution source of truth. Add a strategy-template layer above the raw factor controls: each strategy definition declares its factor set, default weights, search space, eligibility filters, risk flags, and user-facing explanation. Extend the factor engine from the current four hard-coded technical factors into a backward-compatible strategy-aware scoring engine, with optional fundamentals for the value-quality strategy and clear degradation when free data is unavailable.

**Tech Stack:** FastAPI, Pydantic, pandas, Bootstrap, pytest, existing mootdx/yfinance data adapters, existing Phase 3.1 parallel optimizer.

---

## 1. Product Decisions

### 1.1 What Phase 3.2 Adds

- A first-class strategy library for portfolio stock selection.
- Initial strategy templates:
  - `steady_low_vol_momentum`: 稳健低波动动量策略
  - `strong_trend_breakout`: 强趋势突破策略
  - `high_liquidity_trend`: 高流动性趋势策略
  - `drawdown_control_rotation`: 回撤控制型轮动策略
  - `value_quality`: 价值质量因子策略
- A WebUI strategy selector above raw factor controls.
- Strategy-specific default factor weights and search spaces.
- Strategy-specific diagnostics explaining why a stock was selected or skipped.
- Strategy-aware factor optimization: choose a template, then optimize parameters inside that template.
- Backward compatibility with Phase 3.1 custom factor blend behavior.

### 1.2 What Phase 3.2 Does Not Promise

- No claim that any strategy has a guaranteed high win rate.
- No real brokerage execution.
- No hidden AI buy/sell decisions.
- No full institutional factor platform, industry-neutral optimizer, or Barra-style risk model.
- No fundamental-data completeness guarantee; free data availability remains a hard constraint.

### 1.3 Strategy Library Philosophy

The library should make the workflow more like:

```text
Choose selection strategy template
-> Review its factor logic and default assumptions
-> Optimize its allowed parameters on train/validation windows
-> Apply a candidate parameter set
-> Rerun normal rolling portfolio backtest
-> Use latest selected stocks for paper-trading/manual review
```

It should not make the workflow:

```text
Pick a magic strategy name
-> Trust it blindly
-> Buy automatically
```

---

## 2. Initial Strategy Definitions

### 2.1 稳健低波动动量策略

**Intent:** Prefer stocks with positive medium-term momentum, controlled realized volatility, adequate liquidity, and confirmed trend.

**Core factors:**

- `momentum_return`: higher is better
- `realized_volatility`: lower is better
- `downside_volatility`: lower is better
- `liquidity_turnover`: higher is better
- `ma_trend`: higher is better

**Default behavior:**

- Monthly rebalance works well as the default.
- Volatility and downside volatility carry negative weights.
- Use risk flags for high volatility, large recent drawdown, and insufficient trend quality.

### 2.2 强趋势突破策略

**Intent:** Prefer stocks breaking above recent ranges with volume confirmation and strong price trend.

**Core factors:**

- `breakout_strength`: higher is better
- `momentum_return`: higher is better
- `volume_expansion`: higher is better
- `ma_trend`: higher is better
- `realized_volatility`: moderate penalty when too high

**Default behavior:**

- Weekly or biweekly rebalance may be useful, but monthly remains allowed.
- Skip symbols that do not have enough history for breakout lookback.
- Add risk flags for high volatility breakout and failed liquidity confirmation.

### 2.3 高流动性趋势策略

**Intent:** Prefer large, liquid names whose trend is positive enough to trade with lower liquidity/slippage risk.

**Core factors:**

- `liquidity_turnover`: higher is better
- `volume_stability`: higher is better
- `ma_trend`: higher is better
- `momentum_return`: higher is better
- `realized_volatility`: lower is mildly better

**Default behavior:**

- Strong liquidity filters before ranking.
- Useful for users who care about executable paper-trading candidates.
- Risk flags should highlight underinvested or illiquid results.

### 2.4 回撤控制型轮动策略

**Intent:** Rotate into stocks with acceptable momentum but explicitly avoid names with recent deep drawdowns or unstable downside behavior.

**Core factors:**

- `momentum_return`: higher is better
- `max_drawdown_window`: lower is better
- `downside_volatility`: lower is better
- `recovery_strength`: higher is better
- `liquidity_turnover`: higher is better

**Default behavior:**

- Penalize recent drawdown more heavily than raw momentum.
- Prefer smoother validation curves in optimization.
- Risk flags should emphasize drawdown and too few selected stocks.

### 2.5 价值质量因子策略

**Intent:** Prefer stocks with better valuation/quality characteristics, then use technical confirmation to avoid value traps.

**Core factors, when data exists:**

- `pe_inverse`: higher is better
- `pb_inverse`: higher is better
- `roe`: higher is better
- `revenue_growth`: higher is better
- `profit_growth`: higher is better
- `ma_trend`: higher is better as confirmation
- `liquidity_turnover`: higher is better as tradability filter

**Free-data caveat:**

- `mootdx` K-line data does not provide these fundamentals.
- `yfinance` may provide some A-share fundamentals for `.SS` / `.SZ` tickers, but coverage can be incomplete.
- Phase 3.2 must implement value-quality as a graceful-degradation strategy:
  - if enough fundamentals exist, use them;
  - if not, mark symbols with `missing_fundamentals`;
  - if coverage is too low, surface a strategy-level warning instead of silently pretending quality factors were used.

---

## 3. Backend Design

### 3.1 New Strategy Models

Create `portfolio_selection_strategy_models.py` with:

```python
class StrategyFactorSpec(BaseModel):
    key: str
    label: str
    direction: Literal["higher_better", "lower_better"]
    default_weight: float
    default_lookback: int | None = None
    lookback_candidates: list[int] = Field(default_factory=list)
    weight_candidates: list[float] = Field(default_factory=list)
    required: bool = True


class PortfolioSelectionStrategyDefinition(BaseModel):
    strategy_id: str
    name: str
    description: str
    suitable_for: str
    caveats: list[str] = Field(default_factory=list)
    default_rebalance_frequency: Literal["weekly", "biweekly", "monthly"] = "monthly"
    factors: list[StrategyFactorSpec]
    default_top_n: int = 5
    top_n_candidates: list[int] = Field(default_factory=lambda: [3, 5, 10])
    score_threshold_candidates: list[float | None] = Field(default_factory=lambda: [None])
```

Add a request-side config:

```python
class PortfolioSelectionStrategyConfig(BaseModel):
    strategy_id: str = "custom_factor_blend"
    enabled: bool = True
    parameter_overrides: dict[str, Any] = Field(default_factory=dict)
```

Add an optional field to `PortfolioBacktestRequest`:

```python
selection_strategy: PortfolioSelectionStrategyConfig | None = None
```

Backward compatibility rule:

- If `selection_strategy` is missing or `strategy_id == "custom_factor_blend"`, keep Phase 3.1 scoring behavior.

### 3.2 Strategy Library Module

Create `portfolio_selection_strategy_library.py`:

```python
def list_selection_strategies() -> list[PortfolioSelectionStrategyDefinition]:
    ...

def get_selection_strategy(strategy_id: str) -> PortfolioSelectionStrategyDefinition:
    ...

def build_factor_search_space_for_strategy(
    strategy: PortfolioSelectionStrategyDefinition,
) -> FactorSearchSpace:
    ...
```

The library should be deterministic and pure. No market data fetches should happen inside it.

### 3.3 Strategy-Aware Factor Engine

Keep current `score_candidates(...)` as the legacy wrapper. Add a strategy-aware path:

```python
def score_candidates_with_strategy(
    data_by_symbol: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp | str,
    selection_config: SelectionConfig,
    strategy_config: PortfolioSelectionStrategyConfig,
    fundamentals_by_symbol: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ...
```

Add a superset factor calculator:

```python
def calculate_strategy_factor_values(
    data: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    strategy: PortfolioSelectionStrategyDefinition,
    selection_config: SelectionConfig,
    fundamentals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ...
```

Technical factor keys to support in Phase 3.2:

- `momentum_return`
- `realized_volatility`
- `downside_volatility`
- `liquidity_turnover`
- `ma_trend`
- `breakout_strength`
- `volume_expansion`
- `volume_stability`
- `max_drawdown_window`
- `recovery_strength`

Fundamental factor keys to support when available:

- `pe_inverse`
- `pb_inverse`
- `roe`
- `revenue_growth`
- `profit_growth`

### 3.4 Fundamentals Adapter

Create `portfolio_fundamentals.py`:

```python
@dataclass(frozen=True)
class FundamentalsBundle:
    data_by_symbol: dict[str, dict[str, Any]]
    providers: dict[str, str]
    warnings: list[str]
    diagnostics: dict[str, Any]


def load_portfolio_fundamentals(
    symbols: list[str],
    data_provider: str = "auto",
) -> FundamentalsBundle:
    ...
```

MVP behavior:

- Start with yfinance-derived fundamentals when available.
- Cache results in memory during one optimization/backtest run.
- Never fail the whole backtest solely because fundamentals are unavailable.
- Add warnings and diagnostics for coverage percentage.

### 3.5 Strategy-Aware Optimization

Extend Phase 3.1 optimization:

- `PortfolioFactorOptimizationRequest` may include `strategy_id` or use `base_request.selection_strategy`.
- If no explicit raw `search_space` is provided, generate search space from selected strategy.
- Trial result must include `strategy_id`, `strategy_name`, and strategy diagnostics.
- Existing raw factor optimization remains valid.

---

## 4. WebUI Design

### 4.1 Portfolio Workbench Controls

Add near the current `选股因子` / `因子优化` area:

- `portfolioSelectionStrategy`
- `portfolioSelectionStrategyDescription`
- `portfolioSelectionStrategyCaveats`
- `portfolioApplyStrategyDefaultsButton`
- `portfolioStrategyFactorSummary`

Default selected strategy:

```text
稳健低波动动量策略
```

Keep an advanced option:

```text
自定义因子组合
```

### 4.2 Optimization Controls

When strategy changes:

- update default factor weights;
- update search-space candidate inputs;
- update default Top N candidates;
- update caveats;
- do not automatically run optimization;
- do not erase user edits without confirmation if fields have been manually changed.

### 4.3 Result Rendering

Optimization result rows should show:

- strategy name
- objective score
- train annual return
- validation annual return
- validation volatility
- validation downside volatility
- validation trend R-squared
- validation max drawdown
- turnover
- Top N
- factor parameter summary
- strategy risk flags
- missing-data warnings
- `应用参数`

### 4.4 Safety Copy

Use short labels:

- `策略模板`
- `参数候选`
- `虚拟盘参考`
- `不会自动下单`
- `基本面覆盖不足`

Avoid language implying guaranteed success or direct buy recommendations.

---

## 5. API Shape

### 5.1 List Strategy Library

```http
GET /portfolio-selection-strategies
```

Response:

```json
{
  "strategies": [
    {
      "strategy_id": "steady_low_vol_momentum",
      "name": "稳健低波动动量策略",
      "description": "...",
      "default_top_n": 5,
      "factors": [...]
    }
  ]
}
```

### 5.2 Backtest Request

Existing `/portfolio-backtest/jobs` accepts:

```json
{
  "selection_strategy": {
    "strategy_id": "steady_low_vol_momentum",
    "enabled": true,
    "parameter_overrides": {}
  }
}
```

### 5.3 Optimization Request

Existing `/portfolio-factor-optimization/jobs` accepts:

```json
{
  "base_request": {
    "...": "...",
    "selection_strategy": {
      "strategy_id": "strong_trend_breakout",
      "enabled": true
    }
  },
  "search_space": null,
  "max_trials": 200,
  "max_workers": 8
}
```

If `search_space` is null, backend uses the strategy library default search space.

---

## 6. Detailed Task List

### 6.0 子任务总览与 Todo Board

Phase 3.2 should be implemented as twelve small, testable tasks. The task order matters: models and strategy definitions come first, strategy factor calculation comes next, then backtest/optimization integration, then API/UI, and finally documentation plus browser smoke.

**Execution order:**

```text
Task 1-3: Strategy contract and library
-> Task 4-6: Strategy factor calculation and scoring
-> Task 7-8: Backtest and optimization integration
-> Task 9-11: API and WebUI
-> Task 12: Documentation and end-to-end smoke
```

**Workstream A: Strategy Contract And Library**

- [x] Task 1: Add strategy models and request compatibility.
  - [x] Define `StrategyFactorSpec`.
  - [x] Define `PortfolioSelectionStrategyDefinition`.
  - [x] Define request-side `PortfolioSelectionStrategyConfig`.
  - [x] Add optional `selection_strategy` to `PortfolioBacktestRequest`.
  - [x] Preserve old Phase 3.1 request payload compatibility.
  - [x] Add model tests for valid definitions, invalid directions, and old payloads.
  - [x] Commit the model-only change.
- [x] Task 2: Build the initial strategy library.
  - [x] Add `steady_low_vol_momentum`.
  - [x] Add `strong_trend_breakout`.
  - [x] Add `high_liquidity_trend`.
  - [x] Add `drawdown_control_rotation`.
  - [x] Add `value_quality`.
  - [x] Add `custom_factor_blend` as the backward-compatible escape hatch.
  - [x] Add `list_selection_strategies()`.
  - [x] Add `get_selection_strategy(strategy_id)`.
  - [x] Test uniqueness, required fields, factor definitions, and missing strategy errors.
  - [x] Commit the library definition change.
- [x] Task 3: Generate optimization search spaces from strategy templates.
  - [x] Convert factor lookback candidates into optimizer candidates.
  - [x] Convert factor weight candidates into optimizer candidates.
  - [x] Enforce `Top N` candidates within 1-20.
  - [x] Preserve Phase 3.1 raw factor optimization behavior.
  - [x] Add value-quality search-space coverage for fundamentals plus technical confirmation.
  - [x] Test every strategy can generate a valid default search space.
  - [x] Commit the strategy-search-space change.

**Workstream B: Strategy Factor Engine And Scoring**

- [x] Task 4: Extend the factor engine with the strategy factor superset.
  - [x] Add momentum-return calculation that is safe for rolling rebalance dates.
  - [x] Add realized volatility.
  - [x] Add downside volatility.
  - [x] Add liquidity turnover.
  - [x] Add moving-average trend confirmation.
  - [x] Add breakout strength.
  - [x] Add volume expansion.
  - [x] Add volume stability.
  - [x] Add recent max drawdown.
  - [x] Add recovery strength.
  - [x] Return clear `skip_reason` for insufficient history.
  - [x] Add synthetic-data tests for each new factor.
  - [x] Add lookahead-safety tests.
  - [x] Commit the factor-engine change.
- [x] Task 5: Add strategy-aware candidate scoring.
  - [x] Normalize each factor cross-sectionally per rebalance date.
  - [x] Apply `higher_better` and `lower_better` directions correctly.
  - [x] Apply strategy default weights.
  - [x] Apply request-level parameter overrides.
  - [x] Return strategy factor values in candidate diagnostics.
  - [x] Return normalized factor values in candidate diagnostics.
  - [x] Keep legacy `FactorConfig` scoring unchanged.
  - [x] Test steady low-vol momentum ranks smooth uptrends above jagged names.
  - [x] Test strong trend breakout requires breakout plus volume confirmation.
  - [x] Test drawdown-control rotation penalizes recent deep drawdowns.
  - [x] Commit the strategy-scoring change.
- [x] Task 6: Add optional fundamentals support for value-quality.
  - [x] Add a `FundamentalsBundle` model.
  - [x] Add a yfinance-backed loader with no mandatory network dependency in tests.
  - [x] Add dependency injection or mock hooks for unit tests.
  - [x] Calculate `pe_inverse`.
  - [x] Calculate `pb_inverse`.
  - [x] Read or derive `roe` when available.
  - [x] Read or derive revenue growth when available.
  - [x] Read or derive profit growth when available.
  - [x] Add coverage diagnostics for loaded/missing fundamentals.
  - [x] Warn clearly when value-quality coverage is too low.
  - [x] Ensure missing fundamentals never crash a backtest.
  - [x] Commit the fundamentals change.

**Workstream C: Portfolio Backtest And Optimization Integration**

- [x] Task 7: Integrate strategy scoring into portfolio backtest and stock-pool scan.
  - [x] Detect `request.selection_strategy`.
  - [x] Route `custom_factor_blend` and missing strategy config to legacy scoring.
  - [x] Route named strategies to strategy-aware scoring.
  - [x] Include `strategy_id` in scan diagnostics.
  - [x] Include factor diagnostics in candidate rankings.
  - [x] Include strategy warnings in result warnings.
  - [x] Preserve automatic `60/00` universe behavior.
  - [x] Test named strategy and legacy paths.
  - [x] Commit the backtest integration change.
- [x] Task 8: Integrate the strategy library with factor optimization.
  - [x] Allow optimization requests to omit raw `search_space` when a named strategy is selected.
  - [x] Resolve the selected strategy before generating trials.
  - [x] Generate trials from the strategy search space.
  - [x] Preserve `max_workers <= 8`.
  - [x] Preserve process/thread backend selection.
  - [x] Include strategy id/name in trial results.
  - [x] Include strategy warnings and risk flags in optimization output.
  - [x] Keep default ranking by validation smooth-uptrend objective.
  - [x] Test strategy-derived optimization and legacy raw optimization.
  - [x] Commit the optimizer integration change.

**Workstream D: API And WebUI**

- [x] Task 9: Add strategy library API.
  - [x] Add `GET /portfolio-selection-strategies`.
  - [x] Return all five requested strategies.
  - [x] Return labels, descriptions, caveats, defaults, and factors.
  - [x] Keep existing portfolio endpoints unchanged.
  - [x] Add API tests.
  - [x] Commit the API change.
- [x] Task 10: Add strategy selector to the WebUI.
  - [x] Add selector above the raw factor controls.
  - [x] Default to `稳健低波动动量策略`.
  - [x] Provide `自定义因子组合` for legacy behavior.
  - [x] Render strategy description.
  - [x] Render caveats and data limitations.
  - [x] Render factor summary.
  - [x] Add an explicit apply-defaults button.
  - [x] Include `selection_strategy` in portfolio backtest requests.
  - [x] Ensure changing strategy does not silently erase user edits.
  - [x] Add template/JS tests.
  - [x] Commit the WebUI selector change.
- [x] Task 11: Add strategy-aware optimization UI.
  - [x] Include selected strategy in optimization requests.
  - [x] Allow strategy defaults to populate candidate lists.
  - [x] Display strategy name in optimization results.
  - [x] Display validation annual return, max drawdown, turnover, volatility, trend score, and overfitting risk.
  - [x] Display missing fundamentals warnings for value-quality.
  - [x] Preserve `应用参数` behavior.
  - [x] Avoid any wording that implies guaranteed returns.
  - [x] Add template/JS tests.
  - [x] Commit the optimization UI change.

**Workstream E: Documentation, Verification, And Handoff**

- [ ] Task 12: Documentation and E2E smoke.
  - [ ] Update README with the strategy-library concept.
  - [ ] Explain raw factors vs strategy templates vs factor optimization.
  - [ ] Document all five initial strategies.
  - [ ] Document value-quality free-data limitations.
  - [ ] Document that outputs are virtual/paper-trading references, not automatic orders.
  - [ ] Run focused backend tests.
  - [ ] Run template tests.
  - [ ] Run the full pytest suite.
  - [ ] Run `git diff --check`.
  - [ ] Start a local FastAPI server.
  - [ ] Browser-smoke the five strategy options.
  - [ ] Browser-smoke a tiny strategy-aware optimization.
  - [ ] Browser-smoke applying one optimized result back into normal portfolio backtest.
  - [ ] Confirm holdings, rebalance, candidates, trades, warnings, and diagnostics render.
  - [ ] Commit the documentation and smoke-tested final state.

**Definition of Done for Phase 3.2:**

- [ ] Users can pick a strategy template before running portfolio selection backtests.
- [ ] The five initial strategies have real factor definitions, defaults, caveats, and optimization ranges.
- [ ] Named strategy backtests use strategy-aware scoring.
- [ ] `custom_factor_blend` keeps the existing Phase 3.1 raw-factor behavior.
- [ ] Optimization can run from strategy defaults without forcing users to hand-edit raw candidate lists.
- [ ] Value-quality handles missing fundamentals honestly and visibly.
- [ ] Frontend requests, backend API models, optimizer results, and rendered diagnostics all use the same strategy id/name.
- [ ] Tests and browser smoke pass before declaring the prototype usable.

### Task 1: Add Strategy Models And Backward-Compatible Request Field

**Files:**

- Create: `portfolio_selection_strategy_models.py`
- Modify: `portfolio_models.py`
- Test: `test/test_portfolio_selection_strategy_models.py`
- Test: `test/test_portfolio_models.py`

**Step 1: Write failing tests**

- Assert a valid `PortfolioSelectionStrategyDefinition` serializes to JSON.
- Assert invalid factor direction fails.
- Assert `PortfolioBacktestRequest` accepts `selection_strategy`.
- Assert old `PortfolioBacktestRequest` payloads without `selection_strategy` still validate.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_portfolio_selection_strategy_models.py test/test_portfolio_models.py -q
```

Expected: FAIL because the models and field do not exist.

**Step 3: Implement**

- Add strategy model classes.
- Add optional `selection_strategy` to `PortfolioBacktestRequest`.
- Keep default request behavior unchanged.

**Step 4: Verify**

```bash
python -m pytest test/test_portfolio_selection_strategy_models.py test/test_portfolio_models.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add portfolio_selection_strategy_models.py portfolio_models.py test/test_portfolio_selection_strategy_models.py test/test_portfolio_models.py
git commit -m "feat: add portfolio selection strategy models"
```

### Task 2: Build Strategy Library Definitions

**Files:**

- Create: `portfolio_selection_strategy_library.py`
- Test: `test/test_portfolio_selection_strategy_library.py`

**Step 1: Write failing tests**

- Assert the five requested strategy ids exist.
- Assert strategy ids are unique.
- Assert each strategy has at least three factors.
- Assert each factor has non-empty label, direction, default weight, and candidate search values.
- Assert `custom_factor_blend` exists for backward compatibility.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_portfolio_selection_strategy_library.py -q
```

Expected: FAIL because the library does not exist.

**Step 3: Implement**

- Define the five strategy templates.
- Add `custom_factor_blend`.
- Add `list_selection_strategies()`.
- Add `get_selection_strategy(strategy_id)`.
- Add strategy-not-found error with clear message.

**Step 4: Verify**

```bash
python -m pytest test/test_portfolio_selection_strategy_library.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add portfolio_selection_strategy_library.py test/test_portfolio_selection_strategy_library.py
git commit -m "feat: add portfolio selection strategy library"
```

### Task 3: Generate Search Space From Strategy Templates

**Files:**

- Modify: `portfolio_selection_strategy_library.py`
- Modify: `portfolio_factor_optimization_models.py`
- Test: `test/test_portfolio_selection_strategy_library.py`
- Test: `test/test_portfolio_factor_optimization_models.py`

**Step 1: Write failing tests**

- Assert each strategy can generate a non-empty optimization search space.
- Assert generated Top N candidates are within 1-20.
- Assert generated weights are finite.
- Assert `value_quality` search space includes technical confirmation plus fundamentals-related candidates.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_portfolio_selection_strategy_library.py test/test_portfolio_factor_optimization_models.py -q
```

Expected: FAIL because strategy-to-search-space mapping does not exist.

**Step 3: Implement**

- Add `build_factor_search_space_for_strategy(...)`.
- Support raw Phase 3.1 `FactorSearchSpace` generation for compatible factors.
- Add placeholder mapping for strategy-only factors that will be consumed by the strategy-aware scorer in later tasks.

**Step 4: Verify**

```bash
python -m pytest test/test_portfolio_selection_strategy_library.py test/test_portfolio_factor_optimization_models.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add portfolio_selection_strategy_library.py portfolio_factor_optimization_models.py test/test_portfolio_selection_strategy_library.py test/test_portfolio_factor_optimization_models.py
git commit -m "feat: derive optimization search space from selection strategies"
```

### Task 4: Extend Factor Engine With Strategy Factor Superset

**Files:**

- Modify: `factor_engine.py`
- Test: `test/test_factor_engine.py`
- Test: `test/test_selection_strategy_factor_engine.py`

**Step 1: Write failing tests**

Using synthetic OHLCV data:

- Assert `breakout_strength` is higher for a new-high breakout series.
- Assert `volume_expansion` is higher when recent volume expands.
- Assert `max_drawdown_window` is worse for a deep-drop series.
- Assert `downside_volatility` is lower for a smooth uptrend.
- Assert lookahead safety uses only bars before the rebalance date.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_factor_engine.py test/test_selection_strategy_factor_engine.py -q
```

Expected: FAIL because the factor superset does not exist.

**Step 3: Implement**

- Add `calculate_strategy_factor_values(...)`.
- Keep `calculate_symbol_factors(...)` unchanged for legacy Phase 3.1 behavior.
- Add helper functions for rolling drawdown, downside volatility, breakout, volume expansion, volume stability, and recovery strength.
- Return `skip_reason` for insufficient history.

**Step 4: Verify**

```bash
python -m pytest test/test_factor_engine.py test/test_selection_strategy_factor_engine.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add factor_engine.py test/test_factor_engine.py test/test_selection_strategy_factor_engine.py
git commit -m "feat: calculate strategy-aware selection factors"
```

### Task 5: Add Strategy-Aware Candidate Scoring

**Files:**

- Modify: `factor_engine.py`
- Modify: `selection_engine.py` if needed
- Test: `test/test_selection_strategy_scoring.py`

**Step 1: Write failing tests**

- Build three synthetic symbols.
- For `steady_low_vol_momentum`, assert smooth low-vol uptrend ranks above jagged high-return series.
- For `strong_trend_breakout`, assert breakout + volume confirmation ranks above non-breakout.
- For `drawdown_control_rotation`, assert deep recent drawdown is penalized.
- Assert legacy `score_candidates(...)` output is unchanged for existing FactorConfig path.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_selection_strategy_scoring.py test/test_factor_engine.py -q
```

Expected: FAIL because strategy-aware scoring does not exist.

**Step 3: Implement**

- Add `score_candidates_with_strategy(...)`.
- Normalize factors by direction:
  - higher better: `(value - min) / (max - min)`
  - lower better: `(max - value) / (max - min)`
- Apply strategy factor weights.
- Emit `strategy_factor_values`, `normalized_strategy_factors`, `strategy_id`, and `skip_reason`.

**Step 4: Verify**

```bash
python -m pytest test/test_selection_strategy_scoring.py test/test_factor_engine.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add factor_engine.py selection_engine.py test/test_selection_strategy_scoring.py test/test_factor_engine.py
git commit -m "feat: score candidates with selection strategy templates"
```

### Task 6: Add Fundamentals Adapter For Value-Quality Strategy

**Files:**

- Create: `portfolio_fundamentals.py`
- Modify: `factor_engine.py`
- Test: `test/test_portfolio_fundamentals.py`
- Test: `test/test_selection_strategy_scoring.py`

**Step 1: Write failing tests**

- Assert mocked fundamentals produce `pe_inverse`, `pb_inverse`, `roe`, `revenue_growth`, and `profit_growth`.
- Assert missing fundamentals do not crash scoring.
- Assert low coverage creates warnings and diagnostics.
- Assert `value_quality` ranks a high-quality mocked symbol above a low-quality mocked symbol when data is available.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_portfolio_fundamentals.py test/test_selection_strategy_scoring.py -q
```

Expected: FAIL because no fundamentals adapter exists.

**Step 3: Implement**

- Add `FundamentalsBundle`.
- Add yfinance-backed loader with strict timeout/error capture.
- Add test injection hooks so tests do not hit network.
- Add coverage diagnostics:
  - `requested_symbols`
  - `loaded_fundamentals`
  - `coverage_pct`
  - `missing_symbols`
- Add `missing_fundamentals` skip/warning logic for value-quality scoring.

**Step 4: Verify**

```bash
python -m pytest test/test_portfolio_fundamentals.py test/test_selection_strategy_scoring.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add portfolio_fundamentals.py factor_engine.py test/test_portfolio_fundamentals.py test/test_selection_strategy_scoring.py
git commit -m "feat: load optional fundamentals for value quality strategy"
```

### Task 7: Integrate Strategy Scoring Into Portfolio Backtest And Scan

**Files:**

- Modify: `portfolio_backtest_runner.py`
- Modify: `universe_scan_runner.py`
- Test: `test/test_portfolio_backtest_runner.py`
- Test: `test/test_universe_scan_runner.py`

**Step 1: Write failing tests**

- Assert a backtest request with `selection_strategy.strategy_id` uses strategy-aware scoring.
- Assert old requests without `selection_strategy` still use legacy scoring.
- Assert `candidate_rankings` include strategy factor columns.
- Assert scan diagnostics include `strategy_id` and strategy warnings.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_portfolio_backtest_runner.py test/test_universe_scan_runner.py -q
```

Expected: FAIL because portfolio runners do not yet branch by selection strategy.

**Step 3: Implement**

- In backtest/scan runners, detect `request.selection_strategy`.
- Use legacy `score_candidates(...)` for `custom_factor_blend`.
- Use `score_candidates_with_strategy(...)` for named strategies.
- Include strategy diagnostics in `scan_diagnostics`.

**Step 4: Verify**

```bash
python -m pytest test/test_portfolio_backtest_runner.py test/test_universe_scan_runner.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add portfolio_backtest_runner.py universe_scan_runner.py test/test_portfolio_backtest_runner.py test/test_universe_scan_runner.py
git commit -m "feat: run portfolio backtests with selection strategies"
```

### Task 8: Integrate Strategy Library With Factor Optimization

**Files:**

- Modify: `portfolio_factor_optimization_models.py`
- Modify: `portfolio_factor_optimizer.py`
- Test: `test/test_portfolio_factor_optimization_models.py`
- Test: `test/test_portfolio_factor_optimizer.py`

**Step 1: Write failing tests**

- Assert optimization request can omit raw search space when strategy id is provided.
- Assert strategy default search space is used.
- Assert trial result includes strategy id/name.
- Assert custom raw Phase 3.1 optimization still works.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_portfolio_factor_optimization_models.py test/test_portfolio_factor_optimizer.py -q
```

Expected: FAIL because optimization still assumes raw `FactorSearchSpace`.

**Step 3: Implement**

- Allow `search_space` to be optional when strategy is present.
- Resolve strategy before candidate generation.
- Generate candidate configs from strategy defaults.
- Include strategy metadata in diagnostics/result payload.

**Step 4: Verify**

```bash
python -m pytest test/test_portfolio_factor_optimization_models.py test/test_portfolio_factor_optimizer.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add portfolio_factor_optimization_models.py portfolio_factor_optimizer.py test/test_portfolio_factor_optimization_models.py test/test_portfolio_factor_optimizer.py
git commit -m "feat: optimize portfolio selection strategy templates"
```

### Task 9: Add Strategy Library API

**Files:**

- Modify: `main.py`
- Test: `test/test_portfolio_api.py`

**Step 1: Write failing tests**

- Assert `GET /portfolio-selection-strategies` returns all five requested strategy ids.
- Assert response contains name, description, caveats, default Top N, and factor specs.
- Assert existing portfolio endpoints still exist.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_portfolio_api.py -q
```

Expected: FAIL because the API route does not exist.

**Step 3: Implement**

- Add route `GET /portfolio-selection-strategies`.
- Return deterministic JSON from `list_selection_strategies()`.

**Step 4: Verify**

```bash
python -m pytest test/test_portfolio_api.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add main.py test/test_portfolio_api.py
git commit -m "feat: expose portfolio selection strategy library api"
```

### Task 10: Add Strategy Selector To WebUI

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`

**Step 1: Write failing tests**

- Assert DOM ids:
  - `portfolioSelectionStrategy`
  - `portfolioSelectionStrategyDescription`
  - `portfolioSelectionStrategyCaveats`
  - `portfolioApplyStrategyDefaultsButton`
  - `portfolioStrategyFactorSummary`
- Assert JS functions:
  - `loadPortfolioSelectionStrategies`
  - `renderPortfolioSelectionStrategyOptions`
  - `applyPortfolioSelectionStrategyDefaults`
  - `collectPortfolioSelectionStrategyConfig`
- Assert request payload includes `selection_strategy`.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_index_template.py -q
```

Expected: FAIL because no strategy selector exists.

**Step 3: Implement**

- Add compact strategy selector near `选股因子`.
- Load strategies from `/portfolio-selection-strategies`.
- Render explanation and caveats.
- Apply defaults only when user clicks the button.
- Include `selection_strategy` in `collectPortfolioRequest()`.

**Step 4: Verify**

```bash
python -m pytest test/test_index_template.py -q
node - <<'NODE'
const fs = require('fs');
const html = fs.readFileSync('templates/index.html', 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(m => m[1]);
for (const script of scripts) if (script.trim()) new Function(script);
NODE
python -m pytest -q
```

**Step 5: Commit**

```bash
git add templates/index.html test/test_index_template.py
git commit -m "feat: add portfolio selection strategy selector"
```

### Task 11: Strategy-Aware Optimization UI

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`

**Step 1: Write failing tests**

- Assert changing strategy can populate optimization candidate inputs.
- Assert optimization request can omit raw search space only if strategy defaults are selected.
- Assert results display strategy name and strategy-specific warnings.
- Assert no UI copy says strategy guarantees success.

**Step 2: Run tests to verify failure**

```bash
python -m pytest test/test_index_template.py -q
```

Expected: FAIL because optimization UI is not strategy-aware.

**Step 3: Implement**

- Add strategy-aware defaults into `collectPortfolioFactorOptimizationRequest()`.
- Show strategy name in result rows.
- Show missing fundamentals warnings.
- Keep `应用参数` behavior unchanged.

**Step 4: Verify**

```bash
python -m pytest test/test_index_template.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add templates/index.html test/test_index_template.py
git commit -m "feat: optimize portfolio strategy templates from ui"
```

### Task 12: Documentation And E2E Smoke

**Files:**

- Modify: `README.md`
- Modify: `docs/plans/2026-07-08-phase-3-2-selection-strategy-library.md`

**Step 1: Update docs**

- Explain the difference between:
  - raw factors
  - strategy templates
  - factor optimization
  - paper-trading reference
- Document all five initial strategies.
- Document value-quality data coverage limitations.

**Step 2: Run full tests**

```bash
python -m pytest -q
git diff --check
```

Expected: PASS.

**Step 3: Browser smoke**

Start local server:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8005
```

Smoke scenarios:

- Open WebUI.
- Confirm strategy selector loads.
- Select each of the five strategies and confirm description updates.
- Apply defaults for `稳健低波动动量策略`.
- Run tiny manual-pool backtest.
- Run tiny strategy-aware optimization with 4 trials and 2 workers.
- Apply one optimized result.
- Rerun normal portfolio backtest.
- Confirm holdings, rebalance, candidates, trades, warnings, strategy diagnostics, and optimization results render.
- Confirm value-quality strategy shows a clear warning if fundamentals coverage is insufficient.

**Step 4: Commit**

```bash
git add README.md docs/plans/2026-07-08-phase-3-2-selection-strategy-library.md
git commit -m "docs: document phase 3.2 selection strategy library"
```

---

## 7. Acceptance Criteria

- Users can choose one of the five initial stock-selection strategies.
- Each strategy has visible explanation, factor summary, caveats, and defaults.
- Normal portfolio backtests can run with a selected strategy.
- Factor optimization can derive search space from a selected strategy.
- Existing Phase 3.1 custom factor workflow still works.
- Value-quality strategy does not silently fake fundamentals when data is missing.
- API, backend, template, and browser smoke tests pass.
- README explains that strategy results are parameter candidates and virtual/paper-trading references, not live trading instructions.
