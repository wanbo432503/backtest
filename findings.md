# Findings & Decisions

## Requirements
- Correct the single-stock holding-time statistic.
- Replace the hardcoded zero benchmark with a calculated benchmark.
- Return and display current positions, trade history, and equity details.
- Retain the unified strategy engine so every single-stock strategy remains usable in multi-stock signal backtests.
- Preserve existing API fields and optimizer compatibility.

## Research Findings
- `backtest_runner._format_stats()` hardcodes benchmark return to `0.00%`.
- The same formatter labels `final_gross_exposure` as holding time, so a fully exited strategy reports 0% even after many trades.
- `strategy_simulator` already records per-bar `gross_exposure`, positions, trades, equity curve, and signal events.
- `BacktestResult.to_api_response()` currently discards all simulation detail payloads.
- The former backtesting.py path supplied `Buy & Hold Return [%]`, `Exposure Time [%]`, detailed native plots, and more complete statistics.
- The 2026-07-12 unified-engine design intentionally replaced backtesting.py to share strategy and A-share execution semantics across modes.
- Reproducing SH603019 with the MACD strategy produced 694 bars, 68 exposed bars (9.80%), 9 round trips, final exposure 0%, and a raw-close benchmark of 164.52%.
- The existing `renderPortfolioTable()` helper already escapes values, supports custom formatters, responsive overflow, and empty states; it can safely render single-stock positions and trades.
- Signal-portfolio UI already defines the desired position and trade columns, so single-stock reporting can stay consistent without introducing a second presentation system.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Add `exposure_time_pct` to simulator summary | Correct source-level metric reusable by all runners |
| Add benchmark calculation in the single-stock runner | Benchmark depends on the requested instrument data, not portfolio accounting |
| Extend `BacktestResult` additively | Keeps current clients working while enabling complete UI reporting |
| Reuse table styling patterns, not portfolio schema | Avoids tying the single-stock response to portfolio-specific concepts |
| Do not add a second equity table | The full equity curve remains available in the API and is already visualized; duplicating hundreds of rows in the default UI adds noise |

## Issues Encountered
| Issue | Resolution |
|-------|------------|

## Resources
- `backtest_runner.py`
- `strategy_simulator.py`
- `templates/index.html`
- `test/test_backtest_runner.py`
- `test/test_strategy_simulator.py`
- `docs/plans/2026-07-12-unified-strategy-engine-design.md`

## Visual/Browser Findings
- Reported UI shows nine trades and a non-flat equity curve while holding time and benchmark both display 0.00%.
- The chart itself is now responsive after commit `44a3372`; this task must preserve that behavior.
- Browser reproduction with SH603019, MACD volume-divergence strategy, 2023-08-11 through 2026-07-12 now reports strategy return 20.23%, benchmark 164.52%, exposure time 11.24%, and nine completed trades.
- The detail panel distinguishes final exposure 0.00% from period exposure 11.24% and renders all 18 buy/sell order rows.
- Both Bokeh figures remain 556px wide, matching the iframe content width; browser error/warning log is empty.
- Initial detail-panel layout allowed flex shrink to collapse the chart card to zero height; a 680px non-shrinking chart basis keeps the iframe visible and places expanded details below it in the scrollable center pane.
- Independent review found no critical or important issues; its one minor inert-button issue was resolved by rendering single-stock symbols as plain escaped text.
