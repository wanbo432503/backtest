# Phase 3.1 Factor Optimization And Parallel Portfolio Backtest Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable factor optimization for the Phase 3 portfolio selector, raise portfolio Top N to 20, and run many rolling-rebalance portfolio backtests in parallel on an 8-core machine.

**Architecture:** Keep the existing deterministic Phase 3.0 portfolio backtest engine as the source of truth. Add a reusable optimization layer that generates candidate factor/selection configurations, runs the same rolling rebalance backtest across train and validation periods, ranks candidates by validation return with overfit/risk diagnostics, and lets the user apply the winning configuration back to the portfolio workbench. Introduce a bounded parallel worker pool for optimization jobs while reusing loaded universe OHLCV data to avoid refetching the same 60/00 stock pool for every trial.

**Tech Stack:** FastAPI, Pydantic, pandas, concurrent.futures, Bootstrap, pytest, existing mootdx/yfinance data adapters.

---

## 1. Product Decisions

### 1.1 What Phase 3.1 Adds

- `Top N` may be set from 1 to 20.
- The portfolio selector can optimize factor windows, factor weights, score threshold, and optionally Top N.
- Optimization uses the same realistic rolling rebalance logic as normal portfolio backtests:
  - same stock pool
  - same A-share trading rules
  - same rebalance frequency
  - same slippage/commission/lot size
  - same position sizing rules
- Optimization is chronological:
  - train period is used to search and fit candidate factor configurations
  - validation period is used to rank and reject overfit candidates
  - no future bars may be used by factor calculation on any rebalance date
- The default objective is maximum validation annual return, with train-return consistency and risk diagnostics shown beside it.
- The UI must make clear that optimized factors are not live trading signals by themselves; they are parameter candidates for paper-trading or manual review.

### 1.2 What Phase 3.1 Does Not Add

- No real brokerage execution.
- No intraday or high-frequency optimization.
- No AI-generated dynamic buy/sell decisions.
- No industry-neutral optimizer, covariance optimizer, Barra model, or complex portfolio construction.
- No automatic promise that the optimized configuration will work in future markets.

### 1.3 Important Constraint

Phase 3.0 currently reuses `run_portfolio_backtest(...)`, which loads universe OHLCV data inside every run. Phase 3.1 must not call that full data-loading path for every optimization trial. The optimization layer must load data once per job, then run many candidate configurations against the same prepared data bundle.

---

## 2. User Workflow

1. User opens the Phase 3 portfolio workbench.
2. User selects auto `60/00` stock pool, date range, rebalance frequency, risk settings, and Top N up to 20.
3. User opens a new “因子优化” section.
4. User chooses train/validation split, search space, max trials, and worker count. Default worker count is 8.
5. User starts optimization.
6. Backend loads the stock pool and OHLCV data once.
7. Backend generates candidate factor configurations.
8. Backend runs train and validation rolling-rebalance portfolio backtests in parallel.
9. UI shows progress, best validation candidates, risk flags, and parameter sets.
10. User clicks “应用参数” to copy the chosen factor configuration into the normal portfolio backtest form.
11. User reruns normal portfolio backtest with the optimized factors and may use the latest selected stocks for virtual/paper-trading reference.

---

## 3. Optimization Objective

### 3.1 Default Ranking

Each candidate configuration produces two normal portfolio backtests:

- Train backtest: `train_start -> train_end`
- Validation backtest: `validation_start -> validation_end`

Rank candidates by:

```text
objective_score =
validation_annual_return_pct * 1.0
+ min(train_annual_return_pct, validation_annual_return_pct) * 0.15
- abs(validation_max_drawdown_pct) * 0.15
- validation_turnover * 0.02
```

The primary goal remains maximum validation return. The extra terms are small guardrails to avoid selecting a configuration that only wins validation through extreme drawdown or unstable train/validation mismatch.

### 3.2 Diagnostics

Every result must expose:

- train annual return
- validation annual return
- train total return
- validation total return
- train max drawdown
- validation max drawdown
- train/validation return gap
- turnover
- rebalances
- final holdings count
- risk flags:
  - `negative_validation_return`
  - `train_validation_gap`
  - `high_validation_drawdown`
  - `high_turnover`
  - `too_few_rebalances`
  - `too_few_selected_symbols`

---

## 4. Data And Execution Design

### 4.1 Reusable Backtest Context

Create a reusable data context so optimization does not refetch data for every trial:

```python
@dataclass(frozen=True)
class PortfolioBacktestContext:
    data_by_symbol: dict[str, pd.DataFrame]
    providers: dict[str, str]
    warnings: list[str]
    diagnostics: dict[str, Any]
```

Add a lower-level runner:

```python
def run_portfolio_backtest_with_context(
    request: PortfolioBacktestRequest,
    context: PortfolioBacktestContext,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PortfolioBacktestResult:
    ...
```

Then keep the public runner:

```python
def run_portfolio_backtest(request, progress_callback=None):
    context = load_portfolio_backtest_context(request, progress_callback=progress_callback)
    return run_portfolio_backtest_with_context(request, context, progress_callback=progress_callback)
```

### 4.2 Parallelism

Use `concurrent.futures.ProcessPoolExecutor` for CPU-bound optimization workers by default, because pandas scoring/backtesting can be CPU-bound and Python threads are limited by the GIL. Keep the UI and code labels generic as “并行 worker”. Add a fallback `ThreadPoolExecutor` option for environments where process startup or pickling becomes a problem.

Default:

```text
max_workers = min(8, os.cpu_count() or 1)
executor_backend = "process"
```

### 4.3 Job Store

Add a dedicated optimization job store instead of overloading `PortfolioBacktestJobStore`:

- one optimization job may use up to 8 workers internally
- multiple optimization jobs should not run unbounded at the same time
- the store should report progress:
  - generated candidates
  - completed trials
  - failed trials
  - best current objective score
  - current phase

---

## 5. Backend API Shape

### 5.1 Create Optimization Job

```http
POST /portfolio-factor-optimization/jobs
```

Request:

```json
{
  "base_request": { "... PortfolioBacktestRequest ..." },
  "split": {
    "method": "ratio",
    "train_ratio": 0.7,
    "validation_start": null
  },
  "search_space": {
    "momentum_lookback": [20, 40, 60, 90, 120],
    "volatility_lookback": [10, 20, 40],
    "liquidity_lookback": [10, 20, 40],
    "momentum_weight": [0.2, 0.35, 0.5, 0.65],
    "volatility_weight": [-0.5, -0.25, 0],
    "liquidity_weight": [0, 0.15, 0.3],
    "trend_weight": [0, 0.1, 0.2],
    "top_n": [2, 3, 5, 10, 20],
    "score_threshold": [null]
  },
  "max_trials": 200,
  "max_workers": 8,
  "executor_backend": "process",
  "objective": "validation_return_guarded"
}
```

Response:

```json
{
  "job_id": "abc",
  "status": "queued",
  "phase": "queued",
  "message": "等待因子优化",
  "progress": {}
}
```

### 5.2 Poll Optimization Job

```http
GET /portfolio-factor-optimization/jobs/{job_id}
```

Response includes:

```json
{
  "job_id": "abc",
  "status": "running",
  "phase": "optimizing",
  "message": "正在并行回测候选因子",
  "progress": {
    "total_trials": 200,
    "completed_trials": 37,
    "failed_trials": 0,
    "max_workers": 8,
    "best_objective_score": 18.4
  },
  "result": null,
  "error": null
}
```

Final result includes:

```json
{
  "best_result": { "... top candidate ..." },
  "top_results": [{ "... candidate ..." }],
  "split": { "... resolved split dates ..." },
  "diagnostics": { "... data and execution diagnostics ..." },
  "warnings": []
}
```

---

## 6. Task List

### Task 1: Raise Top N Limit To 20

**Files:**

- Modify: `portfolio_models.py`
- Modify: `templates/index.html`
- Modify: `tradingagents_models.py`
- Test: `test/test_portfolio_models.py`
- Test: `test/test_index_template.py`
- Test: `test/test_tradingagents_api.py`

**Step 1: Write failing tests**

- Assert `SelectionConfig(top_n=20)` is valid.
- Assert `SelectionConfig(top_n=21)` fails.
- Assert the UI `portfolioTopN` input allows `max="20"`.
- Assert auto-mode copy no longer says final holdings must be fewer than 5.
- Assert `TradingAgentsPortfolioSummaryRequest` accepts up to 20 selected symbols.

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_portfolio_models.py test/test_index_template.py test/test_tradingagents_api.py -q
```

Expected: FAIL because current limits still reflect the earlier small-portfolio prototype.

**Step 3: Implement**

- Change `SelectionConfig.validate_top_n_cap` from `> 4` to `> 20`.
- Change frontend `portfolioTopN.max` and validation text.
- Keep manual diagnostic pool validation separate: manual pool cannot select more stocks than it contains, but the automatic pool may use Top N up to 20.
- Update AI portfolio summary selected-symbol validation to max 20.

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_portfolio_models.py test/test_index_template.py test/test_tradingagents_api.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add portfolio_models.py templates/index.html tradingagents_models.py test/test_portfolio_models.py test/test_index_template.py test/test_tradingagents_api.py
git commit -m "feat: allow portfolio top n up to 20"
```

---

### Task 2: Extract Reusable Portfolio Backtest Context

**Files:**

- Modify: `portfolio_backtest_runner.py`
- Modify: `universe_scan_runner.py`
- Test: `test/test_portfolio_backtest_runner.py`
- Test: `test/test_universe_scan_runner.py`

**Step 1: Write failing tests**

- Build synthetic OHLCV data for three symbols.
- Create a context once.
- Run two backtests with different factor weights against the same context.
- Assert data loading is not called twice.
- Assert public `run_portfolio_backtest(...)` still returns the same API shape as before.

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_portfolio_backtest_runner.py test/test_universe_scan_runner.py -q
```

Expected: FAIL because no context runner exists yet.

**Step 3: Implement**

- Add `PortfolioBacktestContext`.
- Add `load_portfolio_backtest_context(...)`.
- Add `run_portfolio_backtest_with_context(...)`.
- Refactor `run_portfolio_backtest(...)` into a thin wrapper.
- Preserve existing progress events.

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_portfolio_backtest_runner.py test/test_universe_scan_runner.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add portfolio_backtest_runner.py universe_scan_runner.py test/test_portfolio_backtest_runner.py test/test_universe_scan_runner.py
git commit -m "refactor: reuse portfolio backtest context"
```

---

### Task 3: Add Factor Optimization Models

**Files:**

- Create: `portfolio_factor_optimization_models.py`
- Test: `test/test_portfolio_factor_optimization_models.py`

**Step 1: Write failing tests**

Cover:

- default search space is finite and non-empty
- `max_trials` must be positive
- `max_workers` must be between 1 and 8 by default policy
- split ratio must be between 0.5 and 0.9
- invalid executor backend is rejected
- generated Top N values cannot exceed 20

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimization_models.py -q
```

Expected: FAIL because the module does not exist.

**Step 3: Implement models**

Create:

```python
class OptimizationSplitConfig(BaseModel):
    method: Literal["ratio", "date"] = "ratio"
    train_ratio: float = 0.7
    validation_start: str | None = None

class FactorSearchSpace(BaseModel):
    momentum_lookback: list[int] = Field(default_factory=lambda: [20, 40, 60, 90, 120])
    volatility_lookback: list[int] = Field(default_factory=lambda: [10, 20, 40])
    liquidity_lookback: list[int] = Field(default_factory=lambda: [10, 20, 40])
    momentum_weight: list[float] = Field(default_factory=lambda: [0.2, 0.35, 0.5, 0.65])
    volatility_weight: list[float] = Field(default_factory=lambda: [-0.5, -0.25, 0])
    liquidity_weight: list[float] = Field(default_factory=lambda: [0, 0.15, 0.3])
    trend_weight: list[float] = Field(default_factory=lambda: [0, 0.1, 0.2])
    top_n: list[int] = Field(default_factory=lambda: [2, 3, 5, 10, 20])
    score_threshold: list[float | None] = Field(default_factory=lambda: [None])

class PortfolioFactorOptimizationRequest(BaseModel):
    base_request: PortfolioBacktestRequest
    split: OptimizationSplitConfig = Field(default_factory=OptimizationSplitConfig)
    search_space: FactorSearchSpace = Field(default_factory=FactorSearchSpace)
    max_trials: int = 200
    max_workers: int = 8
    executor_backend: Literal["process", "thread"] = "process"
    objective: Literal["validation_return_guarded"] = "validation_return_guarded"
```

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimization_models.py -q
```

**Step 5: Commit**

```bash
git add portfolio_factor_optimization_models.py test/test_portfolio_factor_optimization_models.py
git commit -m "feat: add portfolio factor optimization models"
```

---

### Task 4: Generate Candidate Factor Configurations

**Files:**

- Create: `portfolio_factor_optimizer.py`
- Test: `test/test_portfolio_factor_optimizer.py`

**Step 1: Write failing tests**

Cover:

- candidate generation is deterministic
- generated candidates are unique
- `max_trials` caps candidate count
- each candidate contains a valid `FactorConfig` and `SelectionConfig`
- Top N never exceeds 20

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimizer.py -q
```

Expected: FAIL because the optimizer module does not exist.

**Step 3: Implement**

Add:

```python
def generate_factor_candidates(
    request: PortfolioFactorOptimizationRequest,
) -> list[PortfolioFactorCandidate]:
    ...
```

Use `itertools.product(...)`, sort deterministically, then cap to `max_trials`.

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimizer.py -q
```

**Step 5: Commit**

```bash
git add portfolio_factor_optimizer.py test/test_portfolio_factor_optimizer.py
git commit -m "feat: generate factor optimization candidates"
```

---

### Task 5: Resolve Train And Validation Windows

**Files:**

- Modify: `portfolio_factor_optimizer.py`
- Test: `test/test_portfolio_factor_optimizer.py`

**Step 1: Write failing tests**

Cover:

- ratio split resolves chronological train and validation dates
- explicit `validation_start` works
- validation start must be after train start and before end date
- both periods must have enough calendar days for at least two rebalance cycles

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimizer.py::test_resolve_optimization_split -q
```

Expected: FAIL because split resolver does not exist.

**Step 3: Implement**

Add:

```python
def resolve_optimization_split(
    base_request: PortfolioBacktestRequest,
    split: OptimizationSplitConfig,
) -> OptimizationSplit:
    ...
```

The train period ends on the day before validation starts. The validation period ends at the base request end date.

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimizer.py -q
```

**Step 5: Commit**

```bash
git add portfolio_factor_optimizer.py test/test_portfolio_factor_optimizer.py
git commit -m "feat: resolve factor optimization split"
```

---

### Task 6: Evaluate One Candidate With Real Rolling Rebalance Logic

**Files:**

- Modify: `portfolio_factor_optimizer.py`
- Test: `test/test_portfolio_factor_optimizer.py`

**Step 1: Write failing tests**

Use synthetic OHLCV where one factor setting clearly wins validation. Assert:

- one candidate runs a train backtest
- the same candidate runs a validation backtest
- returned metrics include train and validation returns
- objective score prioritizes validation annual return
- result includes the exact candidate factor parameters

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimizer.py::test_evaluate_candidate_runs_train_and_validation_backtests -q
```

Expected: FAIL because evaluator does not exist.

**Step 3: Implement**

Add:

```python
def evaluate_factor_candidate(
    candidate: PortfolioFactorCandidate,
    base_request: PortfolioBacktestRequest,
    split: OptimizationSplit,
    context: PortfolioBacktestContext,
) -> PortfolioFactorOptimizationTrialResult:
    ...
```

Build two request copies:

- train request with candidate factors and train dates
- validation request with candidate factors and validation dates

Call `run_portfolio_backtest_with_context(...)` for both.

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimizer.py -q
```

**Step 5: Commit**

```bash
git add portfolio_factor_optimizer.py test/test_portfolio_factor_optimizer.py
git commit -m "feat: evaluate factor candidates"
```

---

### Task 7: Add Parallel Optimization Runner

**Files:**

- Modify: `portfolio_factor_optimizer.py`
- Test: `test/test_portfolio_factor_optimizer.py`

**Step 1: Write failing tests**

Cover:

- runner evaluates all candidates
- runner respects `max_workers`
- progress callback receives total/completed counts
- failed candidate does not crash the whole job if at least one result succeeds
- top results are sorted by objective score descending

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimizer.py::test_run_factor_optimization_parallel -q
```

Expected: FAIL because parallel runner does not exist.

**Step 3: Implement**

Add:

```python
def run_factor_optimization(
    request: PortfolioFactorOptimizationRequest,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PortfolioFactorOptimizationResult:
    ...
```

Implementation order:

1. Load `PortfolioBacktestContext` once using the full base date range.
2. Generate candidates.
3. Resolve split.
4. Submit candidate evaluations to worker pool.
5. Collect results as futures complete.
6. Sort results by objective score.
7. Return best result and top results.

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimizer.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add portfolio_factor_optimizer.py test/test_portfolio_factor_optimizer.py
git commit -m "feat: run factor optimization in parallel"
```

---

### Task 8: Add Optimization Job Store And API

**Files:**

- Create: `portfolio_factor_optimization_progress.py`
- Modify: `main.py`
- Test: `test/test_portfolio_factor_optimization_api.py`
- Test: `test/test_portfolio_factor_optimization_progress.py`

**Step 1: Write failing tests**

Cover:

- `POST /portfolio-factor-optimization/jobs` returns queued job snapshot
- `GET /portfolio-factor-optimization/jobs/{job_id}` returns progress
- failed validation returns 400
- missing job returns 404
- job result serializes best result and top results

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimization_api.py test/test_portfolio_factor_optimization_progress.py -q
```

Expected: FAIL because endpoints and store do not exist.

**Step 3: Implement**

- Add `PortfolioFactorOptimizationJobSnapshot`.
- Add `PortfolioFactorOptimizationJobStore`.
- Add `portfolio_factor_optimization_job_store = PortfolioFactorOptimizationJobStore(run_factor_optimization)`.
- Add:
  - `POST /portfolio-factor-optimization/jobs`
  - `GET /portfolio-factor-optimization/jobs/{job_id}`

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_portfolio_factor_optimization_api.py test/test_portfolio_factor_optimization_progress.py -q
python -m pytest -q
```

**Step 5: Commit**

```bash
git add main.py portfolio_factor_optimization_progress.py test/test_portfolio_factor_optimization_api.py test/test_portfolio_factor_optimization_progress.py
git commit -m "feat: add factor optimization job api"
```

---

### Task 9: Add Frontend Factor Optimization Controls

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`

**Step 1: Write failing tests**

Assert template contains:

- `id="portfolioFactorOptimizationPanel"`
- `id="portfolioOptimizationMaxWorkers"`
- `id="portfolioOptimizationMaxTrials"`
- `id="portfolioOptimizationTrainRatio"`
- `id="startPortfolioFactorOptimizationButton"`
- `function collectPortfolioFactorOptimizationRequest`
- `function createPortfolioFactorOptimizationJob`
- `function pollPortfolioFactorOptimizationJob`

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_index_template.py -q
```

Expected: FAIL because UI controls do not exist.

**Step 3: Implement**

Add an expandable “因子优化” section under “选股因子” or directly after it.

Controls:

- max trials
- max workers, default 8
- train ratio
- executor backend
- optimize Top N checkbox
- Top N candidate list
- factor window candidate lists
- factor weight candidate lists

Keep the UI compact; do not add a landing page or large explanatory copy.

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_index_template.py -q
```

**Step 5: Commit**

```bash
git add templates/index.html test/test_index_template.py
git commit -m "feat: add factor optimization controls"
```

---

### Task 10: Render Optimization Progress And Results

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`

**Step 1: Write failing tests**

Assert template contains:

- `id="portfolioFactorOptimizationProgressPanel"`
- `id="portfolioFactorOptimizationResults"`
- `function renderPortfolioFactorOptimizationProgress`
- `function renderPortfolioFactorOptimizationResults`
- `function applyPortfolioFactorOptimizationResult`

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest test/test_index_template.py -q
```

Expected: FAIL because result rendering does not exist.

**Step 3: Implement**

Display:

- progress bar
- completed / total trials
- worker count
- best current score
- top result table:
  - objective score
  - train annual return
  - validation annual return
  - validation max drawdown
  - turnover
  - Top N
  - factor windows
  - factor weights
  - risk flags
- “应用参数” button per row.

`applyPortfolioFactorOptimizationResult(result)` copies factors and selection values into the normal portfolio form. It must not start a backtest automatically.

**Step 4: Verify**

Run:

```bash
python -m pytest test/test_index_template.py -q
```

**Step 5: Commit**

```bash
git add templates/index.html test/test_index_template.py
git commit -m "feat: render factor optimization results"
```

---

### Task 11: Browser Smoke Test For End-To-End Prototype

**Files:**

- Modify only if failures reveal UI defects.

**Step 1: Start the server**

Run:

```bash
uvicorn main:app --host 127.0.0.1 --port 8005
```

If port 8005 is busy, use 8006.

**Step 2: Browser smoke**

Use the browser testing workflow:

1. Open the local app.
2. Confirm `Top N` accepts `20`.
3. Start a small optimization with:
   - max trials: 4
   - max workers: 2
   - a short synthetic or cached date range
4. Confirm progress appears.
5. Confirm results table appears.
6. Click “应用参数”.
7. Confirm normal factor fields update.
8. Run normal portfolio backtest.
9. Confirm holdings, rebalance, candidates, trades, warnings render.

**Step 3: Fix discovered UI defects**

If any visual or interaction bug appears, write a focused regression test first, then patch.

**Step 4: Verify**

Run:

```bash
python -m pytest -q
```

**Step 5: Commit**

```bash
git add .
git commit -m "fix: polish factor optimization workflow"
```

---

### Task 12: Documentation And Phase 3.1 Completion Checklist

**Files:**

- Modify: `README.md`
- Modify: `docs/plans/2026-07-08-phase-3-1-factor-optimization-parallel-backtest.md`

**Step 1: Update README**

Document:

- Top N may be 1-20
- factor optimization is train/validation based
- default worker count is 8
- results are for research and paper-trading reference, not live execution
- disabling optimization leaves normal portfolio backtest usable

**Step 2: Update this plan**

Mark completed tasks as `[x]`.

**Step 3: Final verification**

Run:

```bash
python -m pytest -q
git diff --check
```

**Step 4: Commit**

```bash
git add README.md docs/plans/2026-07-08-phase-3-1-factor-optimization-parallel-backtest.md
git commit -m "docs: document phase 3.1 factor optimization"
```

---

## 7. Acceptance Criteria

- Top N accepts values from 1 to 20 in backend and frontend.
- Normal portfolio backtest still works without using optimization.
- Factor optimization runs the same rolling rebalance portfolio logic as normal backtest.
- Optimization uses chronological train and validation periods.
- Factor calculations remain lookahead-safe.
- Optimization jobs can use up to 8 parallel workers.
- Optimization loads universe data once per job, not once per trial.
- UI shows progress and final top results.
- User can apply an optimized factor configuration back to the normal portfolio form.
- Full test suite passes.

---

## 8. Risks And Mitigations

- **Overfitting:** Always show train and validation metrics side by side. Rank primarily by validation return and expose risk flags.
- **Data source slowness:** Load OHLCV once per optimization job and reuse context. Keep `max_trials` bounded.
- **CPU pressure:** Default max workers to 8 but allow lower values. Do not run unbounded optimization jobs.
- **Memory pressure:** Limit result payload to top results and compact diagnostics. Avoid returning full equity curves for every trial unless explicitly requested later.
- **Process-pool pickling cost:** Keep process payloads compact. If this becomes too slow, switch `executor_backend` to `thread` or use precomputed factor matrices in a later phase.
- **User confusion between optimized and live-ready:** UI labels should say “应用参数” and “虚拟盘参考”, not “自动买入”.

