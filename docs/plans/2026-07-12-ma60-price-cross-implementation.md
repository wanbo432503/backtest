# MA60 Price Cross Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an MA60 price-cross strategy whose hidden signal strength prioritizes stocks with fewer recent MA60 crossings in multi-stock signal backtests.

**Architecture:** Add one auto-discovered unified strategy module. The module prepares MA values, emits next-open entry/exit intents, and derives entry strength from a fixed trailing crossing count so the existing simulator candidate ordering supplies portfolio priority without UI or portfolio-template changes.

**Tech Stack:** Python, pandas, Pydantic, pytest, the existing unified strategy engine and simulator.

---

### Task 1: Specify MA60 signals and crossing priority

**Files:**
- Create: `test/test_ma60_price_cross_strategy.py`

**Step 1: Write failing unit tests**

Add tests for strict price/MA crossing semantics, trailing up/down crossing count, next-open entry and exit intents, and entry strength equal to the negative crossing count.

**Step 2: Verify RED**

Run: `python -m pytest test/test_ma60_price_cross_strategy.py -q`

Expected: collection fails because `strategies.ma60_price_cross` does not exist.

### Task 2: Implement the unified strategy

**Files:**
- Create: `strategies/ma60_price_cross.py`

**Step 1: Add minimal production code**

Create a strict Pydantic config with `ma_period=60` and `position_pct=0.95`; prepare `ma_value`; count strict sign-changing crosses over the trailing 250 bars; emit entry only on an upward cross and exit only on a downward cross.

**Step 2: Verify GREEN**

Run: `python -m pytest test/test_ma60_price_cross_strategy.py -q`

Expected: all strategy tests pass.

### Task 3: Verify portfolio ordering and registration

**Files:**
- Modify: `test/test_ma60_price_cross_strategy.py`
- Modify: `test/test_strategy_metadata.py`

**Step 1: Write failing integration tests**

Add a multi-symbol simulation test where two symbols signal on the same day and only one position is allowed; assert the lower-cross-count symbol is bought. Add the strategy ID to the explicit registry expectation.

**Step 2: Verify RED, then make the smallest compatibility update**

Run the two test modules, confirm the new expectation fails for the intended reason, and update only the explicit expected strategy set if required.

**Step 3: Verify GREEN and regression suite**

Run focused strategy, simulator, library, metadata, and signal-portfolio tests, then run `python -m pytest -q`.

Expected: zero failures.

### Task 4: Commit and integrate

Stage only the new strategy, tests, and plan documents. Commit the coherent change, fast-forward it into the original workspace, and re-run focused verification there without touching unrelated untracked data.
