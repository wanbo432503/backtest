# Progress Log: Dual-Price Backtesting

## Session: 2026-07-13

### Phase 1: Data and Execution Contract
- **Status:** completed
- Actions taken:
  - Reproduced all major SZ002475 raw price discontinuities from the local mootdx cache.
  - Matched discontinuity dates to mootdx category-1 xdxr events.
  - Verified the 2015 event is a genuine roughly -4.29% move after theoretical ex-right adjustment, not -36.29%.
  - Inspected cache storage and mootdx adjustment implementations.
  - Traced strategy risk, position, valuation, contribution, and fill price assumptions through the simulator.
  - Finalized the frame contract, fail-closed mootdx xdxr behavior, and cash-limited rights subscription behavior.
  - Wrote the implementation design in `docs/plans/2026-07-13-dual-price-backtest-design.md`.
- Files modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `docs/plans/2026-07-13-dual-price-backtest-design.md`

### Phase 2: Failing Contract Tests
- **Status:** completed
- Actions taken:
  - Added adjustment-engine tests for raw preservation, event factors, future-event isolation, and invalid factors.
  - Added simulator contract tests for adjusted signal/raw fill, adjusted stop/raw fill, dividends, bonus shares, valuation, and ex-date limit checks.
  - Added market-data tests for post-cache enrichment and fail-closed `auto` fallback.

### Phase 3: Adjustment Pipeline
- **Status:** completed
- Actions taken:
  - Implemented project-owned xdxr normalization and multiplicative forward-adjustment factors.
  - Kept daily cache content raw and enriched only returned frames.
  - Added mootdx xdxr retrieval, yfinance raw/action fallback, and impossible-gap validation.

### Phase 4: Simulator and Reporting
- **Status:** completed
- Actions taken:
  - Split adjusted strategy prices from raw execution/accounting prices.
  - Applied cash dividends, bonus shares, and cash-limited rights subscriptions before ex-date trading.
  - Corrected raw valuation, realized/unrealized PnL, risk sizing, and limit checks.
  - Preserved dual-price columns through repeated OHLCV preparation.
  - Kept backtesting.py candles and trade markers in adjusted coordinates while returning raw ledger fills.
  - Exposed corporate-action diagnostics in both single-stock and signal-portfolio results.

### Phase 5: Verification and Delivery
- **Status:** completed
- Actions taken:
  - Completed an independent code review and fixed all four Important findings plus the plotting Minor.
  - Added regressions for legitimate large moves, old yfinance caches, Eastmoney raw prices, suspended-stock action dates, and post-bonus plot sizing.
  - Re-ran compilation, full tests, real SZ002475 adjustment, and real MA60 single-stock backtest.

## Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Raw SZ002475 discontinuity scan | Large jumps align with corporate actions | Six 22%-36% jumps align with annual xdxr events | confirmed |
| New dual-price contract tests (RED) | Fail before implementation | Collection fails because `price_adjustment` does not exist | expected failure |
| Provider integration tests (RED) | Cache remains raw and xdxr failure falls back | Missing RawClose; provider remains mootdx | expected failure |
| Focused dual-price suite | All data, simulator, cache, single, portfolio, optimization regressions pass | 106 passed | passed |
| Real SZ002475 2015 regression | Raw -36.29% gap becomes genuine adjusted move | Adjusted return -4.29%; raw cache unchanged | passed |
| Full project suite after review fixes | No regressions | 515 passed, 1 dependency deprecation warning | passed |
| Real MA60 SZ002475 run | Uses backtesting.py chart, raw fills, and action accounting | 26 orders; 2 action events; plot generated | passed |

## Error Log
| Error | Attempt | Resolution |
|-------|---------|------------|
| `NDFrame.fillna(method=...)` TypeError in mootdx `_reversion` | 1 | Rejected built-in implementation |
| KeyError from mootdx `baoli_qfq` positional indexing | 1 | Rejected built-in implementation |
| 2026-07-13 | Missing `test/test_market_data.py` | 1 | Locate the actual market-data test filenames with `rg --files` |
| 2026-07-13 | Hard-coded adjusted return differed by 0.0001 percentage point | 1 | Derive the assertion from the theoretical ex-right reference price |
| 2026-07-13 | Design-doc patch context mismatch | 1 | Re-read the exact file and apply smaller patches |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1, tracing simulator dependencies |
| Where am I going? | Contract tests, adjustment pipeline, simulator integration, verification |
| What's the goal? | Continuous adjusted signals plus accurate raw-price execution/accounting |
| What have I learned? | See `findings.md` |
| What have I done? | Reproduced and classified the discontinuity; selected the data contract direction |
