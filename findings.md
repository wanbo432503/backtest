# Findings & Decisions: Dual-Price Backtesting

## Requirements
- Eliminate corporate-action discontinuities from strategy indicators and charts.
- Preserve actual raw market prices for fills, fees, lot sizing, and cash accounting.
- Credit held positions with bonus shares and cash dividends on ex-dates.
- Prevent false limit-down and stop signals caused by raw ex-right price gaps.
- Keep single-stock, signal-portfolio, optimization, and backtesting.py plotting paths consistent.

## Research Findings
- Cached `SZ002475` mootdx bars are date-complete but raw/unadjusted.
- On 2015-06-19 raw close falls from 53.05 to 33.80 (-36.29%). The xdxr event reports 0.8 cash dividend and 5 bonus shares per 10 shares; theoretical ex-right reference is 35.313, so the genuine close return is about -4.29%.
- Similar raw discontinuities recur in 2016, 2017, 2018, 2019, and 2020 and line up with category-1 xdxr events.
- The current cache metadata explicitly warns that mootdx bars are unadjusted.
- `market_data.fetch_mootdx_ohlcv` returns only raw OHLCV and all strategies, benchmarks, plots, and trading rules consume those same columns.
- mootdx 0.11.7 exposes `Quotes.xdxr`, but its bundled adjustment implementations are incompatible with the installed pandas version.
- Existing daily cache files can remain raw; adjustment should be performed after slicing/loading so corporate actions are recomputed for the requested horizon.
- Strategy code compares `SimulationPosition.entry_price` and `highest_price` with canonical OHLC, so those public values must remain adjusted even though the simulator's internal cost basis is raw.
- Explicit `RiskIntent` stop/target/risk-per-share values are created from canonical strategy bars and therefore belong to adjusted space. Risk-per-share must be divided by the current adjustment factor for raw-cash position sizing.
- All existing positions are closed in full, which permits a single total raw cost basis plus accumulated dividend income per position.
- Current valuation, position rows, contribution rows, and PnL all use canonical `Close`; each must switch to RawClose after positions receive corporate-action share/cash credits.
- `run_single_backtest` calls `prepare_ohlcv` a second time; its former blanket `.title()` conversion changed `RawClose` to `Rawclose`, silently defeating dual-price execution in the single-stock path.
- backtesting.py trade markers need adjusted display prices even though the API trade ledger must retain raw fill prices.
- A fixed 22% rejection threshold is invalid for Beijing Stock Exchange, IPO, and resumed listings; unusual adjusted moves must be warnings rather than universal hard failures.
- Old yfinance cache rows may come from auto-adjusted history, so a provider-specific raw/action contract marker is required without invalidating the much larger mootdx raw cache.
- Corporate actions can occur while an individual stock is suspended; their actual dates must be part of the portfolio calendar and suspended valuation must use the theoretical ex-right reference.
- The real signal-portfolio loader used to reduce every fetched frame to five OHLCV columns. This preserved adjusted signals but silently forced the simulator to recreate `Raw*` from adjusted prices, so portfolio execution and accounting were not actually dual-price.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Add RawOpen/RawHigh/RawLow/RawClose and AdjFactor | Allows signals to remain on adjusted OHLC while execution translates to actual prices |
| Add per-date corporate-action columns | Lets the simulator credit dividends and shares without a separate side channel |
| Use adjusted prices for price-limit eligibility | Adjusted previous/current prices reflect real ex-date returns rather than mechanical gaps |
| Use raw prices for cash, fees, position sizing, PnL, and market value | These quantities must match executable market prices |
| Record corporate-action account events in diagnostics | Makes share/cash changes auditable without pretending they are trades |
| Split internal raw entry cost from public adjusted entry price | Preserves both correct account PnL and existing strategy comparisons |
| Apply dividends before bonus shares on an ex-date | Cash dividend entitlement is based on pre-distribution shares |
| Treat xdxr retrieval as part of the mootdx daily provider contract | Lets `auto` fall back instead of silently accepting an incomplete daily series |
| Preserve old frames with factor 1 | Keeps unit tests and non-mootdx providers compatible while the exact dual-price path remains explicit for mootdx |
| Subscribe to rights shares up to available cash | Prevents negative cash and makes skipped entitlement auditable |
| Preserve dual-price camel-case columns in repeated preparation | Single-stock and portfolio paths can safely reuse the same frame contract |
| Convert raw ledger prices back to adjusted coordinates only for plotting | Keeps backtesting.py markers aligned with adjusted candles without corrupting the trade ledger |
| Use a yfinance provider-contract cache marker instead of bumping the global cache schema | Refreshes ambiguous yfinance data without discarding reusable mootdx raw history |
| Treat post-adjustment large moves as audit warnings | Avoids rejecting legitimate 30% or unrestricted trading sessions |
| Schedule actions independently from K-line dates | Credits suspended holdings on the legal action date and keeps cash available to the rest of the portfolio |
| Preserve the entire dual-price contract at the portfolio ingress | Ensures universe scanning can use adjusted OHLC while the shared simulator still receives raw execution prices and action metadata |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Built-in qfq helpers fail under current pandas | Implement the small required calculation locally with regression tests |

## Resources
- `market_data.py`
- `market_data_cache.py`
- `strategy_simulator.py`
- `backtest_runner.py`
- `a_share_rules.py`
- `docs/plans/2026-07-13-dual-price-backtest-design.md`
- mootdx `Quotes.xdxr`
- mootdx `tools/reversion.py` (reference only; incompatible runtime code)

## Visual Findings
- The screenshot shows SZ002475 falling vertically from about 55 to 34 during June 2015 while the comparison chart identifies an ex-right price regime change.
- The application itself displays the warning that mootdx bars are unadjusted; the warning is accurate but insufficient because the raw series is still used for strategy decisions.
