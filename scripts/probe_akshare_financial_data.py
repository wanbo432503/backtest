from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_fundamentals import load_portfolio_fundamentals


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe AkShare financial data structures for A-share symbols.")
    parser.add_argument("symbols", nargs="*", default=["SH600519", "SZ002241"])
    parser.add_argument("--as-of", default="2026-07-08")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    import akshare as ak

    calls: list[tuple[str, Callable[[], Any]]] = [
        ("stock_value_em", lambda: ak.stock_value_em(symbol=_plain_code(args.symbols[0]))),
        ("stock_profit_sheet_by_report_em", lambda: ak.stock_profit_sheet_by_report_em(symbol=args.symbols[0].upper())),
        ("stock_balance_sheet_by_report_em", lambda: ak.stock_balance_sheet_by_report_em(symbol=args.symbols[0].upper())),
        ("stock_cash_flow_sheet_by_report_em", lambda: ak.stock_cash_flow_sheet_by_report_em(symbol=args.symbols[0].upper())),
        ("stock_history_dividend_detail", lambda: ak.stock_history_dividend_detail(symbol=_plain_code(args.symbols[0]), indicator="分红")),
    ]

    print(f"akshare_version={getattr(ak, '__version__', 'unknown')}")
    for name, call in calls:
        _probe_call(name, call, timeout=args.timeout)

    bundle = load_portfolio_fundamentals(
        [symbol.upper() for symbol in args.symbols],
        data_provider="akshare",
        as_of_date=args.as_of,
    )
    print("\n=== derived_factors ===")
    print(bundle.to_diagnostics())
    for symbol, values in bundle.values_by_symbol.items():
        preview = {key: round(value, 6) for key, value in sorted(values.items())}
        print(symbol, preview)


def _probe_call(name: str, call: Callable[[], Any], *, timeout: float) -> None:
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(call).result(timeout=timeout)
    except TimeoutError:
        print(f"\n{name}: TIMEOUT")
        return
    except Exception as exc:
        print(f"\n{name}: ERROR {type(exc).__name__}: {exc}")
        return

    print(f"\n{name}: shape={getattr(result, 'shape', None)}")
    if isinstance(result, pd.DataFrame):
        print("columns=", list(result.columns)[:30])
        if not result.empty:
            print(result.head(2).to_string(max_cols=12))


def _plain_code(symbol: str) -> str:
    normalized = symbol.strip().upper()
    return normalized[2:] if normalized.startswith(("SH", "SZ")) else normalized


if __name__ == "__main__":
    main()
