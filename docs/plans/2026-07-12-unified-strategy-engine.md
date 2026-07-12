# Unified Strategy Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite all seven strategies around one definition and simulation protocol so the same strategy catalog powers both single-stock and multi-stock signal portfolio backtests while the portfolio UI retains dynamic strategy parameter configuration.

**Architecture:** Each strategy module exports one `StrategyDefinition` with a Pydantic config model, UI/optimization metadata, indicator preparation, and bar evaluation. A shared simulator owns order timing, fills, position state, A-share constraints, cash, and equity; single-stock and signal-portfolio runners become thin adapters over that simulator. `/strategies` is the only catalog API and the portfolio UI renders parameters from its metadata.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, pandas, NumPy, Bokeh, vanilla JavaScript, Bootstrap, pytest.

---

### Task 1: Add unified strategy contracts and deterministic library

**Files:**
- Create: `strategy_engine.py`
- Create: `strategy_library.py`
- Modify: `strategy_metadata.py`
- Test: `test/test_strategy_library.py`
- Test: `test/test_strategy_metadata.py`

**Step 1: Write the failing registry tests**

Add tests that define a minimal fake strategy and assert registration, parameter validation, duplicate rejection, default completion, and API serialization:

```python
def test_strategy_library_validates_and_completes_parameters():
    library = StrategyLibrary([FAKE_DEFINITION])

    config = library.validate_config("fake", {"period": 8})

    assert config.period == 8
    assert config.position_pct == 0.95
    assert library.to_catalog()[0]["supported_modes"] == [
        "single_stock",
        "signal_portfolio",
    ]
```

Also assert metadata names exactly match config-model fields and defaults.

**Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest test/test_strategy_library.py test/test_strategy_metadata.py -q
```

Expected: FAIL because `strategy_engine` and `strategy_library` do not exist.

**Step 3: Implement minimal contracts**

Create immutable `RiskIntent`, `EntryIntent`, `ExitIntent`, `StrategyDecision`, `StrategyBarContext`, `StrategyDefinition`, `SimulationPosition`, and `SimulationResult` dataclasses. `StrategyDecision` must return an immutable replacement `next_state` instead of mutating context state. Implement `StrategyLibrary` with explicit `register`, `get`, `list`, `validate_config`, and `to_catalog` methods.

Use these concrete supporting schemas:

```python
@dataclass(frozen=True)
class RiskIntent:
    stop_price: float | None = None
    target_price: float | None = None
    risk_per_share: float | None = None
    risk_budget_pct: float | None = None


@dataclass(frozen=True)
class ExitIntent:
    reason: str
    order_type: Literal["next_open"] = "next_open"


MinHistoryBars = Callable[[BaseModel], int]
```

Use this definition shape:

```python
@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    display_name: str
    description: str
    config_model: type[BaseModel]
    parameters: tuple[StrategyParamMeta, ...]
    prepare_frame: PrepareFrame
    evaluate: EvaluateStrategy
    min_history_bars: MinHistoryBars
    supported_modes: tuple[str, ...] = ("single_stock", "signal_portfolio")
```

Define `StrategyParamMeta` in `strategy_engine.py` and let `strategy_metadata.py` re-export it. During this task, keep the existing static metadata map and query functions so old callers remain green; do not make the generic library import `strategy_metadata.py`, and do not load real project strategies yet.

**Step 4: Run tests to verify GREEN**

Run the Step 2 command. Expected: PASS.

**Step 5: Commit**

```bash
git add strategy_engine.py strategy_library.py strategy_metadata.py test/test_strategy_library.py test/test_strategy_metadata.py
git commit -m "add unified strategy contracts"
```

### Task 2: Rewrite one representative strategy behind a compatibility wrapper

**Files:**
- Rewrite: `strategies/rsi_risk_control.py`
- Test: `test/test_rsi_risk_control_strategy.py`
- Test: `test/test_strategy_library.py`

**Step 1: Write failing RSI definition tests**

Keep existing indicator/helper tests and add tests for the new definition:

```python
def test_rsi_definition_emits_next_open_entry():
    frame = RSI_DEFINITION.prepare_frame(sample_data, RSIConfig())
    context = context_at(frame, entry_index)

    decision = RSI_DEFINITION.evaluate(context)

    assert decision.entry.order_type == "next_open"
    assert decision.entry.suggested_position_pct == 0.95
```

Add an exit test using a filled `SimulationPosition`, cooldown-state behavior, and frame/decision prefix invariance after future rows are appended or changed.

**Step 2: Run tests to verify RED**

```bash
python -m pytest test/test_rsi_risk_control_strategy.py test/test_strategy_library.py -q
```

Expected: FAIL because the RSI module still exports a Backtesting.py class and no definition.

**Step 3: Rewrite RSI strategy**

Add the unified implementation alongside the temporary `Strategy` compatibility wrapper:

- `RSIConfig(BaseModel)` including validators.
- Vectorized `prepare_rsi_frame`.
- Pure `evaluate_rsi(context)` using only context history and position state.
- `STRATEGY_DEFINITION` with the existing ID, display name, description, defaults, and search values.

The temporary Backtesting.py class must delegate indicator and decision helpers to the same pure functions; it must not become a second implementation. Do not change discovery or `main.py` yet, because the other strategy modules do not export definitions and the old runners still require classes.

**Step 4: Run tests to verify GREEN**

Run the Step 2 command. Expected: PASS.

**Step 5: Commit**

```bash
git add strategies/rsi_risk_control.py test/test_rsi_risk_control_strategy.py test/test_strategy_library.py
git commit -m "rewrite rsi strategy for unified engine"
```

### Task 3: Rewrite the remaining fixed-size single-stock strategies

**Files:**
- Rewrite: `strategies/boll_macd_breakout.py`
- Rewrite: `strategies/ma_trend_risk_control.py`
- Rewrite: `strategies/volume_breakout_risk_control.py`
- Rewrite: `strategies/macd_volume_divergence_risk_control.py`
- Test: `test/test_boll_macd_breakout_strategy.py`
- Test: `test/test_ma_trend_risk_control_strategy.py`
- Test: `test/test_volume_breakout_risk_control_strategy.py`
- Test: `test/test_macd_volume_divergence_risk_control_strategy.py`

**Step 1: Add failing definition behavior tests**

For each strategy, add at minimum:

- Config default and invalid cross-field validation.
- Indicator preparation without future-row access.
- One qualifying entry producing `next_open`.
- Each materially different exit family already supported by the strategy.
- Suggested position percentage.
- Frame and decision prefix invariance after future rows are appended or changed.

Do not delete existing pure helper tests; reuse their fixtures to prove the definition calls the same rules.

**Step 2: Run tests to verify RED**

```bash
python -m pytest \
  test/test_boll_macd_breakout_strategy.py \
  test/test_ma_trend_risk_control_strategy.py \
  test/test_volume_breakout_risk_control_strategy.py \
  test/test_macd_volume_divergence_risk_control_strategy.py -q
```

Expected: FAIL on missing definitions/config models.

**Step 3: Rewrite the four modules**

For each module, add a Pydantic config, vectorized preparation, pure bar evaluation, metadata, and `STRATEGY_DEFINITION`. Preserve existing formulas, defaults, display names, descriptions, and exit reason strings. Keep a temporary Backtesting.py class that delegates to the pure functions so the old app remains green until Task 6; do not duplicate decision rules in the wrapper.

Position-dependent values such as entry price, breakout line, highest price, holding bars, and cooldown come from `StrategyBarContext.position` and `context.state`; do not hide them in global/module state.

**Step 4: Run tests to verify GREEN**

Run Step 2. Expected: PASS.

**Step 5: Commit**

```bash
git add strategies test/test_boll_macd_breakout_strategy.py test/test_ma_trend_risk_control_strategy.py test/test_volume_breakout_risk_control_strategy.py test/test_macd_volume_divergence_risk_control_strategy.py
git commit -m "rewrite fixed-size strategies for unified engine"
```

### Task 4: Rewrite ATR sizing and add Pin Bar to the common library

**Files:**
- Rewrite: `strategies/ma_breakout_atr_risk_control.py`
- Create: `strategies/trend_pullback_pin_bar.py`
- Modify: `strategy_library.py`
- Modify: `strategy_metadata.py`
- Test: `test/test_ma_breakout_atr_risk_control_strategy.py`
- Create: `test/test_trend_pullback_pin_bar_strategy.py`
- Modify: `test/test_strategy_metadata.py`

**Step 1: Write failing ATR and Pin Bar tests**

Add tests proving:

- ATR strategy preserves strict/bootstrap entry and ATR-based suggested size.
- Pin Bar definition appears in the same library and can run in single-stock mode.
- Pin Bar emits `stop_next_bar` at the pattern high with a one-bar expiry.
- Pin Bar attaches structural/ATR risk information and preserves the two-day trend exit.
- The unified catalog now contains exactly seven dual-mode strategies.
- Appending future bars cannot change prior Pin Bar or ATR decisions.

**Step 2: Run tests to verify RED**

```bash
python -m pytest \
  test/test_ma_breakout_atr_risk_control_strategy.py \
  test/test_trend_pullback_pin_bar_strategy.py \
  test/test_strategy_metadata.py -q
```

Expected: FAIL because Pin Bar is not in the library and ATR still uses Backtesting.py state.

**Step 3: Implement both definitions**

Move Pin Bar indicator, strength, entry, risk-plan, and trend-exit logic out of `signal_portfolio_runner.py`. Define `TrendPullbackPinBarConfig` without market breadth fields; those become a portfolio overlay in Task 7.

Rewrite ATR strategy with the same common contracts. Ensure both strategies return a suggested size but never inspect portfolio cash.

At this point all seven modules export valid definitions. Add fail-fast discovery and a cached `get_strategy_library()` provider to `strategy_library.py`, then switch `strategy_metadata.py` query functions from the static map to the fully populated library. Keep the old Backtesting.py registry in `main.py` unchanged until Task 6.

**Step 4: Run tests to verify GREEN**

Run Step 2. Expected: PASS.

**Step 5: Commit**

```bash
git add strategies strategy_library.py strategy_metadata.py test/test_ma_breakout_atr_risk_control_strategy.py test/test_trend_pullback_pin_bar_strategy.py test/test_strategy_metadata.py
git commit -m "add atr and pin bar unified strategies"
```

### Task 5: Build the shared single/multi-symbol simulator

**Files:**
- Create: `strategy_simulator.py`
- Test: `test/test_strategy_simulator.py`

**Step 1: Write failing execution-lifecycle tests**

Use small deterministic OHLCV frames and fake definitions to cover one behavior per test:

```python
def test_close_signal_fills_at_next_open_without_lookahead(): ...
def test_stop_next_bar_only_fills_when_next_high_reaches_trigger(): ...
def test_pending_order_expires_after_declared_bars(): ...
def test_t1_blocks_same_day_exit(): ...
def test_shared_cash_allocates_stronger_signal_first(): ...
def test_max_positions_and_gross_exposure_are_hard_caps(): ...
def test_lot_rounding_and_trade_costs_are_applied(): ...
def test_strategy_suggested_size_is_capped_by_portfolio_limit(): ...
def test_drawdown_gate_is_evaluated_before_pending_entries(): ...
def test_same_bar_stop_and_target_uses_conservative_stop_first(): ...
def test_strategy_state_is_replaced_not_mutated_in_place(): ...
```

Add a lookahead sentinel test: mutate rows after the evaluated date and assert the earlier decision and fill remain unchanged.

**Step 2: Run tests to verify RED**

```bash
python -m pytest test/test_strategy_simulator.py -q
```

Expected: FAIL because the simulator does not exist.

**Step 3: Implement the minimal simulator**

Implement a calendar loop with these explicit phases:

```python
for date in calendar:
    update_holding_bars()
    execute_protective_and_pending_exits(date)
    update_drawdown_and_entry_gate(date)
    execute_pending_entries(date)
    evaluate_strategies_at_close(date)
    record_equity(date)
```

Keep strategy evaluation pure. Store pending orders, fills, positions, immutable per-symbol state replacements, contributions, rejected-order diagnostics, and equity inside a simulation state object. Define suggested size and risk budget as fractions of pre-fill equity. Reuse `a_share_rules.py` for execution constraints. Apply protective risk from T+1, use adverse open pricing for stop gaps, target pricing for favorable target gaps, and stop-first priority when one bar touches both.

**Step 4: Run tests to verify GREEN**

Run Step 2. Expected: PASS.

**Step 5: Commit**

```bash
git add strategy_simulator.py test/test_strategy_simulator.py
git commit -m "add shared strategy simulator"
```

### Task 6: Migrate single-stock backtest and optimization

**Files:**
- Rewrite: `backtest_runner.py`
- Modify: `optimization_runner.py`
- Modify: `optimization_progress.py`
- Modify: `main.py`
- Rewrite: `strategies/*.py` to remove temporary Backtesting.py wrappers
- Modify: `strategy_metadata.py`
- Test: `test/test_backtest_runner.py`
- Test: `test/test_optimization_runner.py`
- Test: `test/test_optimize_api.py`
- Test: `test/test_optimization_progress.py`
- Test: `test/test_strategy_metadata.py`

**Step 1: Write failing runner tests**

Update tests to pass `StrategyLibrary`, not `dict[str, type[Strategy]]`. Assert:

- Parameter validation occurs before simulation.
- One-symbol simulation preserves the legacy response keys.
- Plot HTML contains price/equity output and buy/sell markers without calling `Backtest`.
- Metrics still use `annual_return_pct * 0.4 + sharpe * 0.3 - abs(max_drawdown_pct) * 0.3`.
- Optimizer train/validate runs use the same definition and validated parameters.

**Step 2: Run tests to verify RED**

```bash
python -m pytest \
  test/test_backtest_runner.py \
  test/test_optimization_runner.py \
  test/test_optimize_api.py \
  test/test_optimization_progress.py -q
```

Expected: FAIL because the runner still instantiates `Backtest` classes.

**Step 3: Rewrite the runner**

Call the shared simulator with one symbol and one position slot. Convert `commission` into the single-stock execution-cost policy without adding portfolio-only market breadth filtering. Generate self-contained Bokeh HTML for price, trades, and equity. Keep `BacktestResult.to_api_response()` unchanged.

Change optimizer type hints and calls to accept the library. Do not change the optimization request/response schema.

Atomically switch `main.py` from the legacy class registry to `get_strategy_library()`. After the runner and optimizer no longer require Backtesting.py classes, delete the temporary compatibility wrappers from all strategy modules and remove the old class-inspection loader in the same commit.

In the same change, switch `/strategies` to serialize `StrategyLibrary.to_catalog()` so removing legacy globals never leaves the endpoint broken.

**Step 4: Run tests to verify GREEN**

Run Step 2. Expected: PASS.

**Step 5: Commit**

```bash
git add backtest_runner.py optimization_runner.py optimization_progress.py main.py strategies strategy_metadata.py test/test_backtest_runner.py test/test_optimization_runner.py test/test_optimize_api.py test/test_optimization_progress.py test/test_strategy_metadata.py
git commit -m "migrate single-stock backtests to unified engine"
```

### Task 7: Migrate the signal portfolio request and runner

**Files:**
- Rewrite: `signal_portfolio_models.py`
- Rewrite: `signal_portfolio_runner.py`
- Modify: `portfolio_progress.py`
- Modify: `universe_scan_runner.py`
- Modify: `main.py`
- Test: `test/test_signal_portfolio_models.py`
- Test: `test/test_signal_portfolio_runner.py`
- Test: `test/test_signal_portfolio_api.py`

**Step 1: Write failing multi-strategy portfolio tests**

Parameterize all seven strategy IDs:

```python
@pytest.mark.parametrize("strategy_id", ALL_STRATEGY_IDS)
def test_every_catalog_strategy_runs_in_signal_portfolio(strategy_id):
    result = run_signal_portfolio_with_data(
        request_for(strategy_id),
        two_symbol_data(),
    )
    assert result.config["strategy"]["strategy_name"] == strategy_id
```

Add tests for new nested parameters, unknown strategy/parameter rejection, normalized defaults, legacy flat Pin Bar compatibility, market breadth overlay, shared cash, and strategy diagnostics.

**Step 2: Run tests to verify RED**

```bash
python -m pytest \
  test/test_signal_portfolio_models.py \
  test/test_signal_portfolio_runner.py \
  test/test_signal_portfolio_api.py -q
```

Expected: FAIL because the request and runner are hard-coded to Pin Bar.

**Step 3: Implement generic request validation**

Replace `TrendPullbackPinBarSignalConfig` on the request with:

```python
class SignalPortfolioStrategyConfig(BaseModel):
    strategy_name: str = "trend_pullback_pin_bar"
    parameters: dict[str, Any] = Field(default_factory=dict)
```

Normalize and validate it through `StrategyLibrary`. Add a before-validator for the legacy flat Pin Bar object. Move market breadth thresholds into a portfolio-level `SignalMarketFilterConfig` with an independent `breadth_ma_period=60`. Map legacy breadth fields into this object; reject requests that contain both legacy breadth fields and the new object.

Keep the Pydantic model structurally valid without importing application globals. In `main.py`, after structural model validation and before `signal_portfolio_job_store.submit`, resolve `get_strategy_library()`, validate and complete strategy parameters, and create a normalized request copy. This guarantees unknown strategies and parameters return 400 before a background job is created.

Resolve and validate the strategy before universe loading. Set the effective history requirement to `max(request.selection.min_history_bars, definition.min_history_bars(config))`; update `universe_scan_runner.py` only as needed to accept that normalized requirement.

**Step 4: Rewrite the runner as a simulator adapter**

Delete Pin Bar-specific indicator and execution functions from the runner. Resolve the definition and validated config first, compute the effective history requirement, then load universe data, call the shared simulator, apply the cross-sectional market filter, and map the common result into `SignalPortfolioBacktestResult`.

**Step 5: Run tests to verify GREEN**

Run Step 2. Expected: PASS.

**Step 6: Commit**

```bash
git add signal_portfolio_models.py signal_portfolio_runner.py portfolio_progress.py universe_scan_runner.py main.py test/test_signal_portfolio_models.py test/test_signal_portfolio_runner.py test/test_signal_portfolio_api.py
git commit -m "support all strategies in signal portfolios"
```

### Task 8: Unify the catalog API and dynamic strategy UI

**Files:**
- Modify: `main.py`
- Modify: `templates/index.html`
- Modify: `test/test_strategy_metadata.py`
- Modify: `test/test_index_template.py`
- Modify: `test/test_signal_portfolio_api.py`

**Step 1: Write failing API and template tests**

Assert `/strategies` returns seven strategies with both modes, `engine="unified"`, and the retained compatibility `class_name`. Assert the template contains:

- `id="signalStrategy"`.
- A dedicated dynamic parameter container.
- Shared strategy catalog loading.
- Filtering by `supported_modes.includes('signal_portfolio')`.
- `strategy: { strategy_name, parameters }` in the collected request.
- No hard-coded Pin Bar strategy object or Pin Bar-only parameter collection function.

Keep assertions that the portfolio panel, universe controls, funds, dates, market filter, progress, and results remain.

**Step 2: Run tests to verify RED**

```bash
python -m pytest \
  test/test_strategy_metadata.py \
  test/test_index_template.py \
  test/test_signal_portfolio_api.py -q
```

Expected: FAIL because the page hard-codes Pin Bar and `/strategies` lacks modes.

**Step 3: Update API wiring**

Serialize the already initialized library catalog from `/strategies`, including compatibility and supported-mode fields. Keep `/reload-strategies` only if it can atomically rebuild and validate a complete library; otherwise return a clear deprecation response and remove the unused UI path.

**Step 4: Update the UI**

Load the catalog once and populate both selectors. Extract a reusable parameter-control renderer that accepts a container, strategy metadata, and DOM prefix. Preserve user-entered portfolio-level fields when switching strategies. Collect typed values from metadata and submit the nested strategy object.

**Step 5: Run tests to verify GREEN**

Run Step 2. Expected: PASS.

**Step 6: Commit**

```bash
git add main.py templates/index.html test/test_strategy_metadata.py test/test_index_template.py test/test_signal_portfolio_api.py
git commit -m "add dynamic portfolio strategy selection"
```

### Task 9: Remove obsolete paths, document extension workflow, and verify

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt` to add `bokeh` directly before removing `backtesting` when no production use remains
- Modify or delete: `strategies.json`
- Modify: affected tests under `test/`
- Delete: obsolete strategy metadata or compatibility code only when no caller remains

**Step 1: Document the extension workflow**

Add a concise README section showing how to add one module with config, metadata, preparation, evaluation, and `STRATEGY_DEFINITION`; state that it automatically appears in both single-stock and portfolio UI.

**Step 2: Find obsolete references**

Run:

```bash
rg -n "backtesting|from backtesting import|STRATEGY_CONFIG|TrendPullbackPinBarSignalConfig|_build_signal_frame|strategy_name: 'trend_pullback_pin_bar'" . --glob '!docs/plans/**' --glob '!data/**'
```

Expected: no production strategy implementation or runner-specific Pin Bar branch remains. Test-only compatibility references must be intentional.

Delete `strategies.json` if it is fully superseded by definitions; otherwise replace it with a generated/read-only compatibility artifact and document that it is not a source of truth.

**Step 3: Run focused verification**

```bash
python -m pytest \
  test/test_strategy_library.py \
  test/test_strategy_simulator.py \
  test/test_backtest_runner.py \
  test/test_optimization_runner.py \
  test/test_signal_portfolio_models.py \
  test/test_signal_portfolio_runner.py \
  test/test_signal_portfolio_api.py \
  test/test_index_template.py -q
```

Expected: all pass.

**Step 4: Run the full suite and static checks**

```bash
python -m pytest -q
python -m compileall -q .
git diff --check
```

Expected: all tests pass, compilation exits 0, and no whitespace errors.

**Step 5: Manual smoke test**

Start the server, open the page, and verify:

1. Both strategy selectors contain the same seven strategies.
2. Switching the portfolio strategy changes its parameter controls.
3. A fixed two-stock portfolio job completes for at least RSI and Pin Bar.
4. Single-stock RSI still returns stats and a chart.
5. Strategy validation errors are readable and return 400.

**Step 6: Request code review and fix findings**

Use the `requesting-code-review` skill against the complete diff. Re-run Step 4 after every fix.

**Step 7: Commit**

```bash
git add README.md requirements.txt strategies.json strategy_engine.py strategy_library.py strategy_simulator.py strategies backtest_runner.py signal_portfolio_models.py signal_portfolio_runner.py optimization_runner.py optimization_progress.py portfolio_progress.py main.py templates/index.html test
git commit -m "complete unified single and portfolio strategy engine"
```
