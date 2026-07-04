# Backtest Project Notes

## Project Overview

Backtest is a FastAPI-based stock backtesting application with a browser UI. It supports configurable trading strategies, market data source selection, result charts, compact backtest statistics, and A-share information panels for research reports, capital flow, Dragon Tiger List, and announcements.

The current data-source direction is:

- A-share K-line data should prefer `mootdx` when `data_provider=auto`.
- `yfinance` is the fallback data source when `mootdx` is unavailable.
- Baidu Gushitong is no longer used as a backtest data source.
- A-share search should support both stock codes and Chinese company names where possible.

## Development Requirements

- Keep changes scoped to the requested task and follow existing project patterns.
- Run focused tests after changing backend logic, data adapters, or UI behavior.
- Do not revert unrelated user changes in the working tree.
- Commit work in a timely manner after a coherent change is implemented and verified.
