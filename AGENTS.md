# Backtest Project Notes

## Project Overview

Backtest is a FastAPI-based A-share backtesting application with a browser UI. It supports configurable trading strategies, market data source selection, result charts, compact backtest statistics, and A-share information panels for research reports, capital flow, Dragon Tiger List, and announcements.

The current data-source direction is:

- A-share K-line data should prefer `mootdx` when `data_provider=auto`.
- `yfinance` is only the fallback data source for A-share symbols when `mootdx` is unavailable.
- Baidu Gushitong is no longer used as a backtest data source.
- A-share search should support both stock codes and Chinese company names where possible.
- Backtest targets must remain limited to A-share codes. Do not reintroduce US stocks, HK stocks, or cryptocurrencies as supported backtest symbols.
- Strategy optimization should prioritize fixed, small A-share target pools, usually 1-3 stocks such as `SH603019` and `SZ002241`.
- Do not introduce multi-stock portfolio rotation, weighting, or portfolio-combination complexity unless the user explicitly asks for it.
- Phase 2.0 score uses: `annual_return_pct * 0.4 + sharpe * 0.3 - abs(max_drawdown_pct) * 0.3`.
- Optimization results should emphasize validation score and risk flags over attractive training-only performance.

## Development Requirements

- Keep changes scoped to the requested task and follow existing project patterns.
- Run focused tests after changing backend logic, data adapters, or UI behavior.
- Do not revert unrelated user changes in the working tree.
- Commit work in a timely manner after a coherent change is implemented and verified.
