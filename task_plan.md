# Task Plan: Repair Single-Stock Backtest Reporting

## Goal
Restore correct and complete single-stock backtest reporting while retaining the unified strategy simulator shared with signal portfolios.

## Current Phase
Complete

## Phases

### Phase 1: Requirements & Discovery
- [x] Confirm incorrect metric mappings and missing response fields
- [x] Confirm the unified simulator remains the execution engine
- [x] Map existing portfolio result UI patterns for reuse
- **Status:** complete

### Phase 2: Contract Tests
- [x] Add failing tests for exposure time and benchmark return
- [x] Add failing tests for positions, trades, and equity response payloads
- [x] Add failing template tests for single-stock detail rendering
- **Status:** complete

### Phase 3: Backend Implementation
- [x] Compute summary exposure and benchmark metrics
- [x] Preserve simulation detail payloads in BacktestResult
- [x] Keep optimization and legacy response compatibility
- **Status:** complete

### Phase 4: UI Implementation
- [x] Render current positions and trade history
- [x] Render clear empty states and benchmark caveat
- [x] Preserve responsive single-stock chart layout
- **Status:** complete

### Phase 5: Verification & Delivery
- [x] Run focused and full automated tests
- [x] Reproduce the reported SH603019 scenario in the browser
- [x] Review diff and prepare the verified change for commit to main
- **Status:** complete

## Key Questions
1. Which existing portfolio result components can be reused without coupling the result schemas?
2. How should benchmark reliability be communicated for unadjusted mootdx prices?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep `strategy_simulator.py` as the execution engine | Maintains identical strategy and A-share order semantics across single-stock and signal-portfolio modes |
| Restore reporting instead of reintroducing backtesting.py | Avoids two execution engines and semantic drift while addressing the actual regressions |
| Define exposure time as percent of curve bars with gross exposure above zero | Matches the former UI label and backtesting.py Exposure Time semantics more closely than final exposure |
| Compute benchmark from the actual backtest data and surface the unadjusted-price caveat | A real benchmark is preferable to a hardcoded zero, but mootdx raw-price limitations must remain visible |
| Add a collapsible single-stock details panel using the existing safe table renderer | Gives current positions and complete trade history without replacing the responsive Bokeh chart |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Four new contract tests failed because the repaired fields and UI did not exist | 1 | Implemented the smallest backend and template changes; focused suite now passes 59 tests |
| Expanded details panel collapsed the flex chart card to zero height | 1 | Added a tested 680px chart-card flex basis/minimum so both chart and details remain accessible by scrolling |
| Review found inert stock-code buttons in the new table | 1 | Rendered single-stock symbols as escaped text instead of unbound buttons |
