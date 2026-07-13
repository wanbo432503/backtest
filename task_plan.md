# Task Plan: Dual-Price A-Share Backtesting

## Goal
Use continuous corporate-action-adjusted prices for signals and charts while preserving raw prices for executions and applying dividends/bonus shares to account holdings.

## Current Phase
Phase 5

## Phases

### Phase 1: Data and Execution Contract
- [x] Reproduce SZ002475 discontinuities
- [x] Confirm mootdx raw bars and xdxr availability
- [x] Trace every simulator price/risk/accounting dependency
- [x] Finalize the dual-price frame contract and fallback behavior
- **Status:** completed

### Phase 2: Failing Contract Tests
- [x] Add tests for adjusted OHLC continuity and raw-column preservation
- [x] Add tests for adjusted signal triggers with raw execution prices
- [x] Add tests for bonus shares, cash dividends, and ex-date limit checks
- [x] Add tests for cache/provider diagnostics and unexplained jumps
- **Status:** completed

### Phase 3: Adjustment Pipeline
- [x] Implement corporate-action normalization and multiplicative factors
- [x] Enrich mootdx daily frames without mutating raw cache files
- [x] Define safe fallback when xdxr data is unavailable
- **Status:** completed

### Phase 4: Simulator and Reporting
- [x] Execute orders at raw prices while evaluating signals on adjusted prices
- [x] Apply held-position corporate actions before ex-date trading
- [x] Keep equity, PnL, plots, benchmarks, and diagnostics internally consistent
- **Status:** completed

### Phase 5: Verification and Delivery
- [x] Run focused and full tests
- [x] Re-run SZ002475 2015 discontinuity and MA60 regression scenarios
- [x] Review diff, commit, and deliver
- **Status:** completed

## Key Questions
1. None currently blocking implementation.

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep provider cache as raw bars and enrich on read | Existing mootdx cache remains reusable and future corporate actions can be reapplied consistently |
| Use multiplicative event factors, not interpolation/clamping | Removes only corporate-action discontinuities and preserves genuine returns |
| Canonical OHLC is adjusted; RawOHLC columns are execution prices | Existing strategies and plots become continuous without rewriting every indicator |
| Filter xdxr events to the requested data end | Avoids incorporating corporate actions after the backtest horizon |
| Do not rely on mootdx built-in adjust helpers | Installed mootdx 0.11.7 helpers fail with the installed pandas API |
| Keep strategy-facing position prices adjusted and internal cost basis raw | Strategies compare entries/stops with adjusted bars, while cash PnL must use executable raw prices |
| Treat mootdx daily xdxr retrieval failure as a provider failure | Never silently run raw discontinuous prices as if they were adjusted; `auto` may fall back to the next provider |
| Subscribe to rights issues only up to available cash | Avoid negative cash while preserving the economic effect and exposing any unsubscribed entitlement in diagnostics |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| mootdx `_reversion` uses removed `fillna(method=...)` | 1 | Use a project-owned tested adjustment engine |
| mootdx `baoli_qfq` uses obsolete integer Series indexing | 1 | Use explicit vectorized/event-indexed calculations |
| Assumed `test/test_market_data.py` existed | 1 | Use `rg --files test` before reading market-data tests |
