# Phase 3.1 Factor Optimization And Parallel Portfolio Backtest Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable factor optimization for the Phase 3 portfolio selector, raise portfolio Top N to 20, and run many rolling-rebalance portfolio backtests in parallel on an 8-core machine.

**Architecture:** Keep the existing deterministic Phase 3.0 portfolio backtest engine as the source of truth. Add a reusable optimization layer that generates candidate factor/selection configurations, runs the same rolling rebalance backtest across train and validation periods, ranks candidates by validation smooth-uptrend quality rather than raw return alone, and lets the user apply the winning configuration back to the portfolio workbench. Introduce a bounded parallel worker pool for optimization jobs while reusing loaded universe OHLCV data to avoid refetching the same 60/00 stock pool for every trial.

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
- The default objective prioritizes a steadily rising validation equity curve: meaningful annual return, lower return volatility, lower downside volatility, lower drawdown, and less train/validation mismatch.
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

Rank candidates by the default `validation_smooth_uptrend` objective:

```text
validation_equity_trend_score =
max(validation_annual_return_pct, 0) * validation_log_equity_trend_r2

objective_score =
validation_equity_trend_score * 0.60
+ validation_annual_return_pct * 0.40
+ min(train_annual_return_pct, validation_annual_return_pct) * 0.10
- validation_return_volatility_pct * 0.35
- validation_downside_volatility_pct * 0.25
- abs(validation_max_drawdown_pct) * 0.25
- validation_turnover * 0.02
```

The most important preference is a validation equity curve that steadily rises. A candidate with slower but smoother gains should be allowed to beat a candidate with higher annual return but violent equity swings. `validation_log_equity_trend_r2` is the R-squared of a linear trend fitted to log equity; a curve that climbs steadily has a higher value, while a jagged curve has a lower value. If validation annual return is negative, the trend score is zero.

### 3.2 Diagnostics

Every result must expose:

- train annual return
- validation annual return
- validation return volatility
- validation downside volatility
- train total return
- validation total return
- train max drawdown
- validation max drawdown
- validation log equity trend R-squared
- validation equity trend score
- validation positive-return day ratio
- train/validation return gap
- turnover
- rebalances
- final holdings count
- risk flags:
  - `negative_validation_return`
  - `train_validation_gap`
  - `high_validation_volatility`
  - `low_equity_trend_quality`
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
  "objective": "validation_smooth_uptrend"
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
    "best_objective_score": 18.4,
    "best_equity_trend_r2": 0.82,
    "best_validation_volatility_pct": 12.5
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

## 6. Detailed Transformation Subtasks And Todolist

This section is the engineering-level checklist for the Phase 3.1 transformation. The later Task List expands these into TDD implementation steps and commits.

### 6.1 Portfolio Size And User Contract

**Purpose:** Turn Phase 3.0 from a tiny “less than 5 holdings” prototype into a configurable portfolio selector while keeping the workflow understandable.

**Todo:**

- [x] Raise backend `SelectionConfig.top_n` cap from 4 to 20.
- [x] Raise frontend `portfolioTopN` max from 4 to 20.
- [x] Update all visible copy that still says “最终持仓少于 5 只”.
- [x] Keep manual diagnostic candidate pools bounded by their actual symbol count.
- [x] Update TradingAgents portfolio summary validation to accept up to 20 selected symbols.
- [x] Add regression tests proving `top_n=20` works and `top_n=21` fails.

**Done when:** Normal portfolio backtest accepts Top N 1-20 in both API and UI, and old single-stock/TradingAgents behavior remains unchanged.

### 6.2 Backtest Engine Refactor For Reusable Data

**Purpose:** Make optimization practical by loading the 60/00 universe once per optimization job instead of once per candidate factor configuration.

**Todo:**

- [x] Introduce `PortfolioBacktestContext` with loaded OHLCV data, providers, warnings, and diagnostics.
- [x] Extract `load_portfolio_backtest_context(...)` from the current `run_portfolio_backtest(...)` loading path.
- [x] Add `run_portfolio_backtest_with_context(...)`.
- [x] Keep `run_portfolio_backtest(...)` as a compatibility wrapper.
- [x] Preserve progress events for normal one-off portfolio backtests.
- [x] Add tests proving two backtests can reuse one context without duplicate data loading.
- [x] Add tests proving the public API response shape is unchanged.

**Done when:** A factor optimizer can run many candidate backtests against the same prepared data bundle, and existing portfolio tests still pass.

### 6.3 Optimization Request And Result Contracts

**Purpose:** Define stable API models before implementing the optimizer, so backend, tests, and frontend speak the same language.

**Todo:**

- [x] Create `portfolio_factor_optimization_models.py`.
- [x] Add `OptimizationSplitConfig`.
- [x] Add `FactorSearchSpace`.
- [x] Add `PortfolioFactorOptimizationRequest`.
- [x] Add candidate/result DTOs for factor values, train metrics, validation metrics, risk flags, and objective score.
- [x] Validate `max_trials > 0`.
- [x] Validate `1 <= max_workers <= 8`.
- [x] Validate split ratio and explicit validation date.
- [x] Validate search-space Top N values do not exceed 20.
- [x] Set default objective to `validation_smooth_uptrend`.

**Done when:** Invalid optimization payloads fail at model validation, and a valid default request can be serialized to JSON without custom encoders.

### 6.4 Candidate Search Space Generation

**Purpose:** Generate deterministic factor configurations for optimization without introducing random or irreproducible behavior.

**Todo:**

- [x] Implement `generate_factor_candidates(...)`.
- [x] Use deterministic Cartesian product over configured windows, weights, Top N, and score threshold.
- [x] Deduplicate equivalent candidates.
- [x] Sort candidates deterministically.
- [x] Respect `max_trials`.
- [x] Include the candidate id/index in each generated candidate.
- [x] Add tests for determinism, uniqueness, and max-trial truncation.

**Done when:** Running candidate generation twice with the same request returns the same ordered candidates.

### 6.5 Train/Validation Split

**Purpose:** Prevent lookahead and overfitting by forcing chronological train and validation windows.

**Todo:**

- [x] Implement `resolve_optimization_split(...)`.
- [x] Support ratio split, default `train_ratio=0.7`.
- [x] Support explicit `validation_start`.
- [x] Ensure train period ends before validation period starts.
- [x] Ensure validation period ends at the base request `end_date`.
- [x] Reject splits that leave too little train or validation data.
- [x] Require enough calendar span for at least two rebalance cycles in both windows.
- [x] Add tests for ratio split, explicit split, and invalid date order.

**Status:** Completed. Added deterministic chronological train/validation splitting with non-overlapping request copies, ratio/date modes, minimum window validation, and result payload metadata for later optimizer summaries.

**Done when:** Every candidate is evaluated on the same resolved train/validation windows, with no overlapping dates.

### 6.6 Smooth-Uptrend Objective Metrics

**Purpose:** Encode the user's preference for slower but steadier equity growth instead of unstable high-return curves.

**Todo:**

- [x] Implement `calculate_equity_curve_quality(...)`.
- [x] Compute annualized validation return volatility.
- [x] Compute annualized validation downside volatility.
- [x] Compute validation log-equity trend R-squared.
- [x] Compute validation positive-return day ratio.
- [x] Compute validation equity trend score.
- [x] Implement `validation_smooth_uptrend` objective score.
- [x] Add risk flags for high volatility and low trend quality.
- [x] Add tests where a smoother rising curve beats a higher-return jagged curve.

**Status:** Completed. Added equity-curve quality metrics, the validation smooth-uptrend objective, and optimization risk flags so ranking can prefer stable upward validation curves over jagged high-return candidates.

**Done when:** Optimizer ranking prefers a stable upward validation equity curve over a high-return but violently fluctuating one, all else reasonably comparable.

### 6.7 Candidate Evaluation With Real Trading Logic

**Purpose:** Ensure optimization is optimizing the actual rolling rebalance portfolio behavior, not a simplified factor-only proxy.

**Todo:**

- [x] Implement `evaluate_factor_candidate(...)`.
- [x] Build a train `PortfolioBacktestRequest` from the base request and candidate factors.
- [x] Build a validation `PortfolioBacktestRequest` from the same candidate factors.
- [x] Run both through `run_portfolio_backtest_with_context(...)`.
- [x] Extract train metrics from the train backtest summary.
- [x] Extract validation metrics and smooth-uptrend quality from validation equity curve.
- [x] Include candidate factor windows, weights, Top N, and score threshold in the result.
- [x] Include compact train/validation risk flags.
- [x] Avoid returning full equity curves for every candidate by default.

**Status:** Completed. Added candidate evaluation that applies each parameter set to train and validation request copies, runs both through the reusable portfolio context, extracts compact metrics, de-duplicates warnings, and returns ranked-trial payloads without full equity curves.

**Done when:** One candidate result explains exactly which parameters were tested and why it ranked where it did.

### 6.8 Parallel Optimization Engine

**Purpose:** Use available CPU cores to evaluate many candidate factor sets quickly while keeping resource use bounded.

**Todo:**

- [x] Implement `run_factor_optimization(...)`.
- [x] Load portfolio context once for the full base date range.
- [x] Generate and resolve candidates before launching workers.
- [x] Support `ProcessPoolExecutor` by default.
- [x] Support `ThreadPoolExecutor` fallback.
- [x] Enforce `max_workers <= 8`.
- [x] Emit progress after each completed trial.
- [x] Keep failed candidate errors without failing the whole job if other candidates succeed.
- [x] Sort top results by objective score descending.
- [x] Limit returned result count to a compact top-N list.

**Status:** Completed. Added the bounded optimization runner with one-time context loading, deterministic candidate preparation, process/thread executor selection, per-trial progress, failure isolation, ranked top results, and compact diagnostics.

**Done when:** A bounded 8-worker optimization run returns ranked candidates with progress updates and does not refetch OHLCV per trial.

### 6.9 Optimization Job API

**Purpose:** Make long-running optimization usable from the browser without blocking the FastAPI request thread.

**Todo:**

- [x] Create `portfolio_factor_optimization_progress.py`.
- [x] Add `PortfolioFactorOptimizationJobSnapshot`.
- [x] Add `PortfolioFactorOptimizationJobStore`.
- [x] Add queued/running/succeeded/failed states.
- [x] Add progress fields for total, completed, failed, best score, trend quality, and volatility.
- [x] Add `POST /portfolio-factor-optimization/jobs`.
- [x] Add `GET /portfolio-factor-optimization/jobs/{job_id}`.
- [x] Return 400 for invalid optimization requests.
- [x] Return 404 for unknown jobs.
- [x] Add API and job-store tests.

**Status:** Completed. Added an independent factor-optimization job store with progress snapshots plus FastAPI create/poll endpoints, request validation errors, missing-job handling, and focused API/job-store regression tests.

**Done when:** The frontend can start an optimization job and poll progress until final ranked results are available.

### 6.10 Frontend Optimization Controls

**Purpose:** Let the user configure optimization without leaving the portfolio workbench.

**Todo:**

- [x] Add an expandable `因子优化` panel.
- [x] Add max trials input.
- [x] Add max workers input, default 8.
- [x] Add train ratio input.
- [x] Add executor backend selector.
- [x] Add search-space controls for factor windows.
- [x] Add search-space controls for factor weights.
- [x] Add optional Top N optimization controls.
- [x] Add validation for worker count and trial count.
- [x] Add `collectPortfolioFactorOptimizationRequest(...)`.
- [x] Add `createPortfolioFactorOptimizationJob(...)`.
- [x] Add `pollPortfolioFactorOptimizationJob(...)`.
- [x] Add template tests for all required DOM ids and functions.

**Status:** Completed. Added compact factor-optimization controls to the portfolio workbench, request collection and validation, backend job creation, polling, and template coverage for the new DOM ids and JS functions.

**Done when:** The user can start a small optimization from the UI using current portfolio workbench settings.

### 6.11 Frontend Results And Apply Flow

**Purpose:** Present optimized factors in a way that supports paper-trading review, not blind live execution.

**Todo:**

- [x] Add optimization progress panel.
- [x] Add optimization result table.
- [x] Show objective score.
- [x] Show train annual return.
- [x] Show validation annual return.
- [x] Show validation volatility.
- [x] Show validation downside volatility.
- [x] Show validation trend R-squared.
- [x] Show validation max drawdown.
- [x] Show turnover.
- [x] Show Top N and factor parameters.
- [x] Show risk flags.
- [x] Add `应用参数` button per result row.
- [x] Implement `applyPortfolioFactorOptimizationResult(...)`.
- [x] Ensure applying parameters does not auto-run a backtest.
- [x] Add UI tests for render and apply functions.

**Status:** Completed. Added progress and result panels, ranked candidate rendering, risk badges, safety copy, and an apply-only flow that writes optimized parameters back into the portfolio form without starting a backtest.

**Done when:** A user can choose an optimized row, copy its parameters into the normal portfolio form, and manually rerun portfolio backtest.

### 6.12 End-To-End Verification

**Purpose:** Prove the prototype is usable from browser and resilient enough for iterative research.

**Todo:**

- [ ] Run focused model tests.
- [ ] Run focused optimizer tests.
- [ ] Run focused API/job-store tests.
- [ ] Run focused template tests.
- [ ] Run full `python -m pytest -q`.
- [ ] Run `git diff --check`.
- [ ] Start local server.
- [ ] Browser-smoke Top N 20.
- [ ] Browser-smoke a tiny optimization with 4 trials and 2 workers.
- [ ] Apply one optimized result.
- [ ] Run normal portfolio backtest with applied parameters.
- [ ] Confirm holdings, rebalance, candidates, trades, warnings, and optimization results render.
- [ ] Update README with user-facing workflow and caveats.

**Done when:** Phase 3.1 can be demonstrated end-to-end without hand-editing requests or relying on hidden backend-only behavior.

### 6.13 WebUI Modification Plan

**Purpose:** Make every Phase 3.1 backend capability discoverable and usable in the existing single-page portfolio workbench. WebUI changes are required for Phase 3.1; backend-only optimization is not considered complete.

**Files:**

- Modify: `templates/index.html`
- Test: `test/test_index_template.py`
- Optional if UI grows too large: create `static/portfolio_optimization.js`
- Optional if CSS grows too large: create `static/portfolio_optimization.css`

**Layout Todo:**

- [x] Keep the first screen as the portfolio workbench; do not add a landing page.
- [x] Keep the normal portfolio backtest controls visible as the primary workflow.
- [x] Add `因子优化` as an expandable panel near `选股因子`, not as a separate page.
- [x] Keep compact operational styling consistent with the current Bootstrap workbench.
- [x] Avoid nested cards inside the existing workbench; use full-width bands, details panels, tables, and compact controls.
- [x] Ensure Top N 1-20 does not break the current two-column form layout.
- [ ] Ensure optimization results remain readable on narrow desktop and mobile widths.

**Form Controls Todo:**

- [x] Add `portfolioFactorOptimizationPanel`.
- [x] Add `portfolioOptimizationMaxTrials`.
- [x] Add `portfolioOptimizationMaxWorkers`.
- [x] Add `portfolioOptimizationTrainRatio`.
- [x] Add `portfolioOptimizationExecutorBackend`.
- [x] Add `portfolioOptimizationIncludeTopN`.
- [x] Add `portfolioOptimizationTopNCandidates`.
- [x] Add text inputs for factor window candidate lists.
- [x] Add text inputs for factor weight candidate lists.
- [x] Add client-side parsing for comma/space separated numeric lists.
- [x] Add validation for max trials, max workers, train ratio, Top N candidates, and empty search-space lists.
- [x] Disable the start button while an optimization job is queued/running.

**Request And State Todo:**

- [x] Reuse `collectPortfolioRequest()` as `base_request`.
- [x] Add `collectPortfolioFactorOptimizationRequest()`.
- [x] Keep `latestPortfolioOptimizationJob` or equivalent client-side state.
- [x] Keep normal portfolio backtest state separate from optimization state.
- [x] Do not mutate normal factor fields until the user clicks `应用参数`.
- [x] Persist no optimization API keys or secrets; this flow does not need secrets.
- [x] Reset stale optimization errors when a new job starts.

**Async Job UX Todo:**

- [x] Add `createPortfolioFactorOptimizationJob()`.
- [x] Add `pollPortfolioFactorOptimizationJob()`.
- [x] Add `renderPortfolioFactorOptimizationProgress()`.
- [x] Show queued/running/succeeded/failed status.
- [x] Show completed trials, failed trials, total trials, worker count, best score, trend R-squared, and volatility.
- [x] Scroll or reveal the optimization progress area after job start.
- [x] Show actionable error text if validation fails or backend job fails.
- [x] Keep the normal “开始组合回测” button usable after optimization failure.

**Result Rendering Todo:**

- [x] Add `portfolioFactorOptimizationProgressPanel`.
- [x] Add `portfolioFactorOptimizationResults`.
- [x] Add `renderPortfolioFactorOptimizationResults()`.
- [x] Render the top optimization candidates in a compact table.
- [x] Display objective score.
- [x] Display train annual return.
- [x] Display validation annual return.
- [x] Display validation volatility.
- [x] Display validation downside volatility.
- [x] Display validation trend R-squared.
- [x] Display validation max drawdown.
- [x] Display turnover.
- [x] Display Top N.
- [x] Display factor windows and weights.
- [x] Display risk flags.
- [x] Provide an empty state before optimization has run.
- [x] Provide a failed state if all candidates fail.

**Apply Parameters Todo:**

- [x] Add `applyPortfolioFactorOptimizationResult(result)`.
- [x] Add an `应用参数` button to each result row.
- [x] Copy selected factor windows into existing portfolio factor inputs.
- [x] Copy selected factor weights into existing portfolio factor inputs.
- [x] Copy selected Top N if the optimized candidate includes Top N.
- [x] Do not automatically start a portfolio backtest after applying parameters.
- [x] Show a small success status after applying parameters.
- [x] Keep the optimized result row visible after applying parameters so the user can compare and rerun manually.

**Safety Copy Todo:**

- [x] Label results as `虚拟盘参考` or `参数候选`, not as automatic live-trading instructions.
- [x] Explain in compact text that smoother validation equity growth is preferred over raw annual return.
- [x] Show train and validation metrics side by side to expose overfitting.
- [x] Show risk flags prominently enough to discourage blindly applying high-risk candidates.
- [x] Avoid large explanatory paragraphs inside the workbench; use terse labels and tooltips where needed.

**WebUI Test Todo:**

- [x] Extend `test/test_index_template.py` to assert every new DOM id exists.
- [x] Assert every new JS function exists.
- [x] Assert the optimization request posts to `/portfolio-factor-optimization/jobs`.
- [x] Assert polling calls `/portfolio-factor-optimization/jobs/{job_id}`.
- [x] Assert `applyPortfolioFactorOptimizationResult` writes back to factor input ids.
- [x] Assert no optimization UI text implies live trading or automatic buying.

**Browser Smoke Todo:**

- [ ] Start the local FastAPI app.
- [ ] Confirm the `因子优化` panel is visible and expandable.
- [ ] Confirm invalid worker/trial inputs show client-side validation.
- [ ] Run a tiny optimization job.
- [ ] Confirm progress updates are visible.
- [ ] Confirm optimization results render.
- [ ] Apply one candidate.
- [ ] Confirm factor fields and Top N update.
- [ ] Run normal portfolio backtest with the applied parameters.
- [ ] Confirm portfolio results still render in the existing result tabs.

**Done when:** A user can configure, start, monitor, review, and apply factor optimization entirely from the WebUI, then manually rerun a normal portfolio backtest with the chosen parameters.

---

## 7. Task List

### Task 1: Raise Top N Limit To 20

**Status:** Completed. Focused RED tests were added first, then backend, WebUI, and TradingAgents limits were updated to support Top N up to 20 while keeping manual diagnostic pools bounded by their actual symbol count.

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

**Status:** Completed. Added `PortfolioBacktestContext`, `load_portfolio_backtest_context(...)`, and `run_portfolio_backtest_with_context(...)`; kept `run_portfolio_backtest(...)` as a compatibility wrapper and added context-reuse tests.

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

**Status:** Completed. Added factor optimization request, split, search-space, candidate, metrics, trial result, and final result models with validation for workers, trials, split dates, objective, and Top N search values.

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
- default objective is `validation_smooth_uptrend`
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
    objective: Literal["validation_smooth_uptrend"] = "validation_smooth_uptrend"
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

**Status:** Completed. Added deterministic candidate generation with duplicate removal, stable candidate ids, max-trial truncation, and valid `FactorConfig` / `SelectionConfig` outputs.

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
- returned metrics include validation return volatility and downside volatility
- returned metrics include validation log-equity trend R-squared
- objective score prefers a smoother rising validation curve over a higher-return but highly volatile curve
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
def calculate_equity_curve_quality(equity_curve: list[dict[str, Any]]) -> dict[str, float]:
    ...

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

`calculate_equity_curve_quality(...)` must compute:

- `return_volatility_pct`: annualized standard deviation of daily equity returns.
- `downside_volatility_pct`: annualized standard deviation of negative daily equity returns.
- `log_equity_trend_r2`: R-squared of a linear fit on log equity over time.
- `positive_return_day_ratio`: ratio of positive equity-return days.
- `equity_trend_score`: `max(annual_return_pct, 0) * log_equity_trend_r2`.

The objective score must use validation quality metrics. It must not let a high-return but jagged validation curve dominate a slower, steadier validation curve when the steadier curve has meaningfully lower volatility and higher trend R-squared.

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
- progress callback exposes best current trend quality and validation volatility when available
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
- `id="portfolioOptimizationExecutorBackend"`
- `id="portfolioOptimizationIncludeTopN"`
- `id="portfolioOptimizationTopNCandidates"`
- `id="startPortfolioFactorOptimizationButton"`
- `function collectPortfolioFactorOptimizationRequest`
- `function createPortfolioFactorOptimizationJob`
- `function pollPortfolioFactorOptimizationJob`
- `fetch('/portfolio-factor-optimization/jobs'`
- `portfolio-factor-optimization/jobs/`

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
Client-side validation must reject invalid worker counts, invalid train ratios, empty candidate lists, and Top N candidates outside 1-20 before posting the job.

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
- `validation_log_equity_trend_r2`
- `validation_return_volatility_pct`
- `validation_downside_volatility_pct`
- `虚拟盘参考`

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
  - validation return volatility
  - validation downside volatility
  - validation trend R-squared
  - validation max drawdown
  - turnover
  - Top N
  - factor windows
  - factor weights
  - risk flags
- “应用参数” button per row.

`applyPortfolioFactorOptimizationResult(result)` copies factors and selection values into the normal portfolio form. It must not start a backtest automatically.
It must show a small success status and keep the optimization results visible for comparison.

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

## 8. Acceptance Criteria

- Top N accepts values from 1 to 20 in backend and frontend.
- Normal portfolio backtest still works without using optimization.
- Factor optimization runs the same rolling rebalance portfolio logic as normal backtest.
- Optimization uses chronological train and validation periods.
- The default objective rewards smooth validation equity growth, not raw annual return alone.
- Results expose validation return volatility, downside volatility, and equity trend quality.
- Factor calculations remain lookahead-safe.
- Optimization jobs can use up to 8 parallel workers.
- Optimization loads universe data once per job, not once per trial.
- UI shows progress and final top results.
- User can apply an optimized factor configuration back to the normal portfolio form.
- Full test suite passes.

---

## 9. Risks And Mitigations

- **Overfitting:** Always show train and validation metrics side by side. Rank by validation smooth-uptrend quality, not return alone; slower but steadier equity growth should beat sharp, unstable gains.
- **Data source slowness:** Load OHLCV once per optimization job and reuse context. Keep `max_trials` bounded.
- **CPU pressure:** Default max workers to 8 but allow lower values. Do not run unbounded optimization jobs.
- **Memory pressure:** Limit result payload to top results and compact diagnostics. Avoid returning full equity curves for every trial unless explicitly requested later.
- **Process-pool pickling cost:** Keep process payloads compact. If this becomes too slow, switch `executor_backend` to `thread` or use precomputed factor matrices in a later phase.
- **User confusion between optimized and live-ready:** UI labels should say “应用参数” and “虚拟盘参考”, not “自动买入”.
