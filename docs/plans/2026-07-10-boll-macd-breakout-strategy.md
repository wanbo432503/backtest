# BOLL MACD Breakout Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a single-stock BOLL upper-band breakout strategy confirmed by a rising middle band and a same-bar MACD golden cross, with optimizable fixed-percentage take-profit and stop-loss parameters.

**Architecture:** Add one strategy module under `strategies/` and rely on the existing dynamic strategy loader. Reuse the existing MACD indicator functions, expose pure entry and risk-price helpers for focused tests, register parameter metadata, and keep the single-stock UI's global risk controls synchronized with strategy parameter inputs so optimization results can be replayed unchanged.

**Tech Stack:** Python, pandas, NumPy, backtesting.py, FastAPI strategy metadata, pytest.

---

### Task 1: Specify signal and exit behavior

**Files:**
- Create: `test/test_boll_macd_breakout_strategy.py`

**Step 1: Write failing entry tests**

Test that entry is true only when the BOLL middle band rises, the close crosses from at-or-below the upper band to above it, and DIF crosses above DEA on the same completed bar. Parameterize the three missing-condition cases.

**Step 2: Write failing risk tests**

Test that `stop_loss_pct` and `take_profit_pct` produce exact boundaries relative to the actual entry price and reject non-positive values.

**Step 3: Run the focused test**

Run: `pytest -q test/test_boll_macd_breakout_strategy.py`

Expected: collection fails because `strategies.boll_macd_breakout` does not exist.

### Task 2: Implement the strategy

**Files:**
- Create: `strategies/boll_macd_breakout.py`
- Test: `test/test_boll_macd_breakout_strategy.py`

**Step 1: Implement indicators and pure decision helpers**

Calculate BOLL middle/upper bands with rolling mean and standard deviation, reuse existing MACD DIF/DEA functions, and implement the exact same-bar entry predicate plus percentage exit predicate.

**Step 2: Implement `BollMACDBreakoutStrategy`**

Use defaults `BOLL(20, 2)`, `MACD(12, 26, 9)`, `stop_loss_pct=1`, `take_profit_pct=1`, and `position_pct=0.95`. Enter only after all three completed-bar conditions, then attach SL/TP orders to the actual fill price after the entry bar so daily A-share backtests begin exit execution on the next trading day.

**Step 3: Run the focused behavior tests**

Run: `pytest -q test/test_boll_macd_breakout_strategy.py`

Expected: behavior tests pass while metadata assertions still fail.

### Task 3: Register optimization metadata

**Files:**
- Modify: `strategy_metadata.py`
- Modify: `test/test_strategy_metadata.py`
- Test: `test/test_boll_macd_breakout_strategy.py`

**Step 1: Add metadata assertions**

Require the strategy API to expose the new name and require both risk parameters to default to `1.0`, use search values `[0.5, 1.0, 1.5, 2.0, 3.0]`, and allow `0.1` through `10.0` in `0.1` increments.

**Step 2: Verify the assertions fail**

Run: `pytest -q test/test_boll_macd_breakout_strategy.py test/test_strategy_metadata.py`

Expected: metadata assertions fail because the strategy has not been added to `STRATEGY_METADATA`.

**Step 3: Add minimal metadata**

Expose fixed default search candidates for BOLL/MACD periods and position size, and the approved five-value search grids for take-profit and stop-loss.

**Step 4: Re-run focused tests**

Run: `pytest -q test/test_boll_macd_breakout_strategy.py test/test_strategy_metadata.py`

Expected: all focused tests pass.

### Task 4: Verify and commit

**Files:**
- Modify: `templates/index.html`
- Modify: `test/test_index_template.py`
- Review all changed files.

**Step 1: Verify UI parameter ownership and risk execution**

Require strategy parameter inputs to remain the submitted source of truth, synchronize matching global risk controls in both directions, use population standard deviation for BOLL, validate the approved risk range and step in the strategy, and run a real `Backtest` regression proving risk orders use actual fill price after the entry bar.

**Step 2: Run related regression tests**

Run: `pytest -q test/test_boll_macd_breakout_strategy.py test/test_strategy_metadata.py test/test_backtest_runner.py test/test_optimization_runner.py`

Expected: all tests pass.

**Step 3: Run the full automated suite**

Run: `pytest -q`

Expected: all tests pass.

**Step 4: Review and commit**

Run: `git diff --check`, inspect `git diff`, then stage only the plan, strategy, metadata, and tests before committing.
