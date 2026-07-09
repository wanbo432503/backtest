from __future__ import annotations

import math
import io
from contextlib import redirect_stderr, redirect_stdout
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from market_data import to_yfinance_symbol


InfoLoader = Callable[[str], dict[str, Any]]
AkshareLoader = Callable[[str], dict[str, pd.DataFrame]]


class FundamentalsBundle(BaseModel):
    values_by_symbol: dict[str, dict[str, float]] = Field(default_factory=dict)
    requested_symbols: list[str] = Field(default_factory=list)
    loaded_symbols: list[str] = Field(default_factory=list)
    missing_symbols: list[str] = Field(default_factory=list)
    coverage_pct: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    errors_by_symbol: dict[str, str] = Field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "requested_symbols": len(self.requested_symbols),
            "loaded_fundamentals": len(self.loaded_symbols),
            "coverage_pct": self.coverage_pct,
            "missing_symbols": self.missing_symbols,
            "errors_by_symbol": self.errors_by_symbol,
            "warnings": self.warnings,
        }


def load_portfolio_fundamentals(
    symbols: list[str],
    *,
    data_provider: str = "auto",
    as_of_date: str | pd.Timestamp | None = None,
    info_loader: InfoLoader | None = None,
    akshare_loader: AkshareLoader | None = None,
    min_coverage_pct: float = 50.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> FundamentalsBundle:
    requested_symbols = list(dict.fromkeys(symbols))
    if not requested_symbols:
        return FundamentalsBundle()

    values_by_symbol: dict[str, dict[str, float]] = {}
    loaded_symbols: list[str] = []
    missing_symbols: list[str] = []
    errors_by_symbol: dict[str, str] = {}
    failed_count = 0

    for symbol in requested_symbols:
        try:
            values = (
                _load_legacy_info_values(symbol, info_loader)
                if info_loader is not None or data_provider == "yfinance"
                else _load_akshare_values(symbol, as_of_date, akshare_loader)
            )
        except Exception as exc:
            values = {}
            errors_by_symbol[symbol] = str(exc)

        if values:
            values_by_symbol[symbol] = values
            loaded_symbols.append(symbol)
        else:
            failed_count += 1
            missing_symbols.append(symbol)
        _emit_progress(
            progress_callback,
            total_symbols=len(requested_symbols),
            loaded_count=len(loaded_symbols),
            failed_count=failed_count,
            current_symbol=symbol,
        )

    coverage_pct = len(loaded_symbols) / len(requested_symbols) * 100
    warnings = []
    if coverage_pct < min_coverage_pct:
        warnings.append(
            f"fundamental coverage low: {coverage_pct:.1f}% "
            f"({len(loaded_symbols)}/{len(requested_symbols)})"
        )
    if data_provider == "mootdx" and loaded_symbols:
        warnings.append("mootdx does not provide fundamentals; akshare fallback was used")

    return FundamentalsBundle(
        values_by_symbol=values_by_symbol,
        requested_symbols=requested_symbols,
        loaded_symbols=loaded_symbols,
        missing_symbols=missing_symbols,
        coverage_pct=coverage_pct,
        warnings=warnings,
        errors_by_symbol=errors_by_symbol,
    )


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    total_symbols: int,
    loaded_count: int,
    failed_count: int,
    current_symbol: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback({
        "phase": "loading_fundamentals",
        "total_symbols": total_symbols,
        "loaded_count": loaded_count,
        "failed_count": failed_count,
        "current_symbol": current_symbol,
    })


def _load_legacy_info_values(
    symbol: str,
    info_loader: InfoLoader | None,
) -> dict[str, float]:
    loader = info_loader or _load_yfinance_info
    yfinance_symbol = to_yfinance_symbol(symbol)
    return _extract_value_quality_factors(loader(yfinance_symbol))


def _load_akshare_values(
    symbol: str,
    as_of_date: str | pd.Timestamp | None,
    akshare_loader: AkshareLoader | None,
) -> dict[str, float]:
    loader = akshare_loader or _load_akshare_financial_frames
    return _extract_akshare_factors(loader(symbol), as_of_date)


@lru_cache(maxsize=256)
def _load_akshare_financial_frames(symbol: str) -> dict[str, pd.DataFrame]:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is not installed") from exc

    plain_code = _plain_a_share_code(symbol)
    eastmoney_symbol = _eastmoney_symbol(symbol)
    return {
        "valuation": _quiet_akshare_call(lambda: ak.stock_value_em(symbol=plain_code)),
        "profit": _quiet_akshare_call(lambda: ak.stock_profit_sheet_by_report_em(symbol=eastmoney_symbol)),
        "balance": _quiet_akshare_call(lambda: ak.stock_balance_sheet_by_report_em(symbol=eastmoney_symbol)),
        "cash_flow": _quiet_akshare_call(lambda: ak.stock_cash_flow_sheet_by_report_em(symbol=eastmoney_symbol)),
        "dividend": _load_akshare_dividend_detail(ak, plain_code),
    }


def _load_akshare_dividend_detail(ak: Any, plain_code: str) -> pd.DataFrame:
    try:
        return _quiet_akshare_call(lambda: ak.stock_history_dividend_detail(symbol=plain_code, indicator="分红"))
    except Exception:
        return pd.DataFrame()


def _quiet_akshare_call(call: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return call()


def _load_yfinance_info(yfinance_symbol: str) -> dict[str, Any]:
    import yfinance as yf

    return dict(yf.Ticker(yfinance_symbol).info or {})


def _extract_value_quality_factors(info: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    pe = _first_finite(info, "trailingPE", "forwardPE")
    pb = _first_finite(info, "priceToBook")
    roe = _first_finite(info, "returnOnEquity")
    revenue_growth = _first_finite(info, "revenueGrowth")
    profit_growth = _first_finite(info, "earningsGrowth", "netIncomeGrowth")

    if pe is not None and pe > 0:
        values["pe_inverse"] = 1 / pe
    if pb is not None and pb > 0:
        values["pb_inverse"] = 1 / pb
    if roe is not None:
        values["roe"] = roe
    if revenue_growth is not None:
        values["revenue_growth"] = revenue_growth
    if profit_growth is not None:
        values["profit_growth"] = profit_growth
    return values


def _extract_akshare_factors(
    frames: dict[str, pd.DataFrame],
    as_of_date: str | pd.Timestamp | None,
) -> dict[str, float]:
    as_of = _as_timestamp(as_of_date)
    values: dict[str, float] = {}

    valuation = _latest_row(frames.get("valuation"), as_of, ("数据日期", "date"))
    market_cap = _row_float(valuation, "总市值", "market_cap")
    close_price = _row_float(valuation, "当日收盘价", "close")
    pe = _row_float(valuation, "PE(TTM)", "PE(静)", "市盈率")
    pb = _row_float(valuation, "市净率", "PB")
    ps = _row_float(valuation, "市销率", "PS")
    pcf = _row_float(valuation, "市现率", "PCF")

    _set_factor(values, "pe_inverse", _inverse_positive(pe))
    _set_factor(values, "pb_inverse", _inverse_positive(pb))
    _set_factor(values, "ps_inverse", _inverse_positive(ps))
    _set_factor(values, "pcf_inverse", _inverse_positive(pcf))

    profit = _latest_row(
        frames.get("profit"),
        as_of,
        ("NOTICE_DATE", "公告日期", "UPDATE_DATE", "REPORT_DATE", "报告期"),
    )
    balance = _latest_row(
        frames.get("balance"),
        as_of,
        ("NOTICE_DATE", "公告日期", "UPDATE_DATE", "REPORT_DATE", "报告期"),
    )
    cash_flow = _latest_row(
        frames.get("cash_flow"),
        as_of,
        ("NOTICE_DATE", "公告日期", "UPDATE_DATE", "REPORT_DATE", "报告期"),
    )

    revenue = _row_float(profit, "OPERATE_INCOME", "TOTAL_OPERATE_INCOME", "营业总收入", "其中：营业收入")
    operating_cost = _row_float(profit, "OPERATE_COST", "营业成本", "其中：营业成本")
    parent_netprofit = _row_float(profit, "PARENT_NETPROFIT", "*归属于母公司所有者的净利润", "归属于母公司所有者的净利润")
    netprofit = _row_float(profit, "NETPROFIT", "*净利润", "净利润")
    total_assets = _row_float(balance, "TOTAL_ASSETS", "ASSET_BALANCE", "*资产合计", "资产合计")
    total_liabilities = _row_float(balance, "TOTAL_LIABILITIES", "LIAB_BALANCE", "*负债合计", "负债合计")
    parent_equity = _row_float(
        balance,
        "TOTAL_PARENT_EQUITY",
        "TOTAL_EQUITY",
        "PARENT_EQUITY_BALANCE",
        "EQUITY_BALANCE",
        "*归属于母公司所有者权益合计",
        "归属于母公司所有者权益合计",
        "所有者权益（或股东权益）合计",
    )
    operating_cashflow = _row_float(
        cash_flow,
        "NETCASH_OPERATE",
        "*经营活动产生的现金流量净额",
        "经营活动产生的现金流量净额",
    )
    capex = _row_float(
        cash_flow,
        "CONSTRUCT_LONG_ASSET",
        "购建固定资产、无形资产和其他长期资产支付的现金",
    )
    dividend_paid = _row_float(
        cash_flow,
        "ASSIGN_DIVIDEND_PORFIT",
        "分配股利、利润或偿付利息支付的现金",
    )

    _set_factor(values, "roe", _safe_divide(parent_netprofit, parent_equity))
    _set_factor(values, "roa", _safe_divide(netprofit or parent_netprofit, total_assets))
    _set_factor(values, "gross_margin", _safe_divide(_subtract(revenue, operating_cost), revenue))
    _set_factor(values, "net_margin", _safe_divide(parent_netprofit, revenue))
    _set_factor(values, "debt_to_assets", _safe_divide(total_liabilities, total_assets))
    _set_factor(values, "operating_cashflow_to_profit", _safe_divide(operating_cashflow, parent_netprofit))
    _set_factor(values, "cashflow_yield", _safe_divide(operating_cashflow, market_cap))
    _set_factor(values, "fcf_yield", _safe_divide(_subtract(operating_cashflow, capex), market_cap))
    _set_factor(values, "dividend_coverage", _safe_divide(operating_cashflow, dividend_paid))

    dividend_values = _extract_dividend_factors(frames.get("dividend"), as_of, close_price)
    values.update(dividend_values)
    return values


def _extract_dividend_factors(
    dividend_frame: pd.DataFrame | None,
    as_of: pd.Timestamp | None,
    close_price: float | None,
) -> dict[str, float]:
    rows = _dated_rows_until(dividend_frame, as_of, ("除权除息日", "公告日期", "实施方案公告日期"))
    if rows.empty:
        return {}

    values: dict[str, float] = {}
    latest = rows.iloc[0]
    cash_per_10_shares = _row_float(latest, "派息", "派息比例")
    dividend_yield = _safe_divide(_safe_divide(cash_per_10_shares, 10), close_price)
    _set_factor(values, "dividend_yield", dividend_yield)

    if as_of is not None:
        cutoff = as_of - pd.DateOffset(years=5)
        recent_rows = rows[rows["_snapshot_date"] >= cutoff]
    else:
        recent_rows = rows.head(5)
    if not recent_rows.empty:
        years = set(recent_rows["_snapshot_date"].dt.year.dropna().astype(int).tolist())
        _set_factor(values, "dividend_stability", min(len(years) / 5, 1.0))
    return values


def _first_finite(info: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = info.get(key)
        numeric_value = _to_float(value)
        if numeric_value is not None:
            return numeric_value
    return None


def _latest_row(
    frame: pd.DataFrame | None,
    as_of: pd.Timestamp | None,
    date_columns: tuple[str, ...],
) -> pd.Series | None:
    rows = _dated_rows_until(frame, as_of, date_columns)
    if not rows.empty:
        return rows.iloc[0]
    if frame is None or frame.empty:
        return None
    return frame.iloc[0]


def _dated_rows_until(
    frame: pd.DataFrame | None,
    as_of: pd.Timestamp | None,
    date_columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    date_column = next((column for column in date_columns if column in frame.columns), None)
    if date_column is None:
        return frame.copy()

    rows = frame.copy()
    rows["_snapshot_date"] = pd.to_datetime(rows[date_column], errors="coerce").dt.normalize()
    rows = rows[rows["_snapshot_date"].notna()]
    if as_of is not None:
        rows = rows[rows["_snapshot_date"] <= as_of]
    return rows.sort_values("_snapshot_date", ascending=False)


def _row_float(row: pd.Series | dict[str, Any] | None, *keys: str) -> float | None:
    if row is None:
        return None
    for key in keys:
        if key not in row:
            continue
        numeric_value = _to_float(row[key])
        if numeric_value is not None:
            return numeric_value
    return None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.replace(",", "").strip()
        if not text or text.lower() in {"false", "nan", "none", "--", "-"}:
            return None
        multiplier = 1.0
        if text.endswith("%"):
            text = text[:-1].strip()
            multiplier = 0.01
        elif text.endswith("亿"):
            text = text[:-1].strip()
            multiplier = 100_000_000.0
        elif text.endswith("万"):
            text = text[:-1].strip()
            multiplier = 10_000.0
        value = text
    else:
        multiplier = 1.0
    try:
        numeric_value = float(value) * multiplier
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric_value):
        return numeric_value
    return None


def _set_factor(values: dict[str, float], key: str, value: float | None) -> None:
    if value is not None and math.isfinite(value):
        values[key] = float(value)


def _inverse_positive(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return 1 / value


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _as_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _plain_a_share_code(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    if normalized.startswith(("SH", "SZ")) and len(normalized) >= 8:
        return normalized[2:]
    if "." in normalized:
        return normalized.split(".")[0]
    return normalized


def _eastmoney_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    if normalized.startswith(("SH", "SZ")):
        return normalized
    code = _plain_a_share_code(symbol)
    prefix = "SH" if code.startswith("6") else "SZ"
    return f"{prefix}{code}"
