# Volume Divergence RSI Long Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add one optimized long-only strategy to the unified library so it is selectable in both single-stock and multi-stock signal portfolio backtests.

**Architecture:** Implement one pure strategy module with a Pydantic config, prefix-safe indicators, close-time decisions, and catalog metadata. Extend the shared simulator with a generic next-bar risk update so the strategy can tighten protective stops without runner-specific branches.

**Tech Stack:** Python 3.13, Pydantic v2, pandas, NumPy, FastAPI, vanilla JavaScript metadata UI, pytest.

---

### Task 1: Add next-bar protective-risk updates

**Files:**
- Modify: `strategy_engine.py`
- Modify: `strategy_simulator.py`
- Test: `test/test_strategy_simulator.py`

1. Write a failing test proving a strategy can return a higher stop after the close and that the simulator applies it from the next bar.
2. Run the focused test and confirm it fails because `StrategyDecision` has no `risk_update`.
3. Add `risk_update: RiskIntent | None` and replace the open position risk after evaluation.
4. Run all simulator tests.

### Task 2: Implement the strategy definition

**Files:**
- Create: `strategies/volume_divergence_rsi_long.py`
- Create: `test/test_volume_divergence_rsi_long_strategy.py`

1. Write failing tests for config validation, prefix-safe preparation, strict entry conditions, initial risk metadata, monotonic trailing stop, two-day MA exit, and unproductive 20-day exit.
2. Confirm RED because the module does not exist.
3. Implement `VolumeDivergenceRSILongConfig`, RSI/MACD/MA/volume/divergence preparation, pure evaluation, metadata, minimum history, and `STRATEGY_DEFINITION`.
4. Run the new strategy tests and simulator tests until GREEN.

### Task 3: Verify catalog and both backtest modes

**Files:**
- Modify: `test/test_strategy_metadata.py`
- Modify: `test/test_signal_portfolio_runner.py`
- Modify: `test/test_index_template.py`

1. Update failing catalog expectations from seven to eight strategies.
2. Assert the new definition advertises both supported modes and is included by the parameterized signal-portfolio runner test.
3. Run strategy catalog, API, UI template, single-stock runner, and signal-portfolio tests.

### Task 4: Document and verify

**Files:**
- Modify: `README.md`

1. Document the new strategy and its exact signal/exit semantics.
2. Run `pytest -q`, `python -m compileall -q .`, JavaScript syntax checking, and `git diff --check`.
3. Commit the completed implementation on `main` without touching `data/`.
