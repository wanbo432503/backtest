from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from a_share_rules import (
    apply_slippage,
    calculate_trade_cost,
    can_buy,
    can_sell,
    round_lot_shares,
)
from factor_engine import score_candidates
from portfolio_data import load_portfolio_ohlcv
from portfolio_models import PortfolioBacktestRequest, PortfolioBacktestResult
from selection_engine import build_rebalance_dates, build_trading_calendar, select_top_candidates
from universe_scan_runner import load_universe_scan_data


@dataclass
class Position:
    symbol: str
    shares: int
    entry_date: str
    holding_bars: int = 0


def run_portfolio_backtest(
    request: PortfolioBacktestRequest,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PortfolioBacktestResult:
    scan_data = load_universe_scan_data(
        request,
        data_loader=load_portfolio_ohlcv,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback({
            "phase": "backtesting",
            "screened_count": len(scan_data.data_by_symbol),
        })
    data_by_symbol = scan_data.data_by_symbol
    calendar = build_trading_calendar(data_by_symbol)
    rebalance_dates = set(
        build_rebalance_dates(calendar, request.start_date, request.end_date, request.rebalance)
    )

    cash = float(request.initial_cash)
    positions: dict[str, Position] = {}
    equity_curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rebalance_events: list[dict[str, Any]] = []
    candidate_rankings: list[dict[str, Any]] = []
    warnings = list(scan_data.warnings)
    scan_diagnostics = dict(scan_data.diagnostics)

    for date in calendar:
        if pd.Timestamp(date) < pd.Timestamp(request.start_date) or pd.Timestamp(date) > pd.Timestamp(request.end_date):
            continue

        day_trades: list[dict[str, Any]] = []
        skipped_trades: list[dict[str, Any]] = []

        if date in rebalance_dates:
            ranking = score_candidates(data_by_symbol, date, request.factors, request.selection)
            selection = select_top_candidates(ranking, request.selection)
            warnings.extend(selection.warnings)
            candidate_rankings.extend(_ranking_rows(date, selection.ranking))
            selected_symbols = [row["symbol"] for row in selection.selected]
            equity_before = _portfolio_value(date, cash, positions, data_by_symbol)
            target_values = _target_values(selected_symbols, equity_before, request)

            sells = _execute_sells(
                date,
                cash,
                positions,
                target_values,
                data_by_symbol,
                request,
                trades,
                day_trades,
                skipped_trades,
            )
            cash = sells
            buys = _execute_buys(
                date,
                cash,
                positions,
                target_values,
                data_by_symbol,
                request,
                trades,
                day_trades,
                skipped_trades,
            )
            cash = buys
            rebalance_events.append({
                "date": _date_str(date),
                "selected_symbols": selected_symbols,
                "trades": day_trades,
                "skipped_trades": skipped_trades,
                "warnings": selection.warnings,
            })

        equity = _portfolio_value(date, cash, positions, data_by_symbol)
        equity_curve.append({
            "date": _date_str(date),
            "equity": round(equity, 6),
            "cash": round(cash, 6),
            "gross_exposure": round(_gross_exposure(date, positions, data_by_symbol) / equity, 6) if equity else 0,
            "drawdown_pct": 0.0,
        })

        for position in positions.values():
            position.holding_bars += 1

    _fill_drawdowns(equity_curve)
    positions_payload = _positions_payload(calendar[-1], positions, data_by_symbol)
    summary = _summary(equity_curve, trades, rebalance_events, request.initial_cash)
    risk_flags = _risk_flags(summary, warnings, rebalance_events)
    scan_diagnostics.update({
        "candidate_ranking_rows": len(candidate_rankings),
        "rebalance_count": len(rebalance_events),
        "selected_symbols": sorted({
            symbol
            for event in rebalance_events
            for symbol in event.get("selected_symbols", [])
        }),
    })

    return PortfolioBacktestResult(
        summary=summary,
        equity_curve=equity_curve,
        positions=positions_payload,
        trades=trades,
        rebalance_events=rebalance_events,
        candidate_rankings=candidate_rankings,
        data_warnings=warnings,
        risk_flags=risk_flags,
        scan_diagnostics=scan_diagnostics,
        config=request.model_dump(mode="json"),
    )


def _target_values(
    selected_symbols: list[str],
    equity: float,
    request: PortfolioBacktestRequest,
) -> dict[str, float]:
    if not selected_symbols:
        return {}
    per_position = equity * request.risk.target_gross_exposure / len(selected_symbols)
    cap = equity * request.risk.max_position_pct
    return {symbol: min(per_position, cap) for symbol in selected_symbols}


def _execute_sells(
    date: pd.Timestamp,
    cash: float,
    positions: dict[str, Position],
    target_values: dict[str, float],
    data_by_symbol: dict[str, pd.DataFrame],
    request: PortfolioBacktestRequest,
    trades: list[dict[str, Any]],
    day_trades: list[dict[str, Any]],
    skipped_trades: list[dict[str, Any]],
) -> float:
    for symbol in list(positions):
        position = positions[symbol]
        row = _row_at(data_by_symbol[symbol], date)
        if row is None:
            continue
        previous_close = _previous_close(data_by_symbol[symbol], date)
        current_value = position.shares * float(row["Close"])
        target_value = target_values.get(symbol, 0.0)
        excess_value = current_value - target_value
        if excess_value <= float(row["Close"]):
            continue

        allowed, reason = can_sell(row.to_dict(), previous_close, position.holding_bars, request.trading)
        if not allowed:
            skipped_trades.append({"symbol": symbol, "side": "sell", "reason": reason})
            continue

        shares = min(position.shares, round_lot_shares(excess_value / float(row["Close"]), request.trading.lot_size))
        if shares <= 0:
            continue
        price = apply_slippage(float(row["Close"]), "sell", request.trading.slippage_pct)
        amount = shares * price
        cost = calculate_trade_cost(amount, "sell", request.trading)
        cash += amount - cost
        position.shares -= shares
        trade = _trade_record(date, symbol, "sell", shares, price, amount, cost, "rebalance")
        trades.append(trade)
        day_trades.append(trade)
        if position.shares <= 0:
            positions.pop(symbol, None)
    return cash


def _execute_buys(
    date: pd.Timestamp,
    cash: float,
    positions: dict[str, Position],
    target_values: dict[str, float],
    data_by_symbol: dict[str, pd.DataFrame],
    request: PortfolioBacktestRequest,
    trades: list[dict[str, Any]],
    day_trades: list[dict[str, Any]],
    skipped_trades: list[dict[str, Any]],
) -> float:
    for symbol, target_value in target_values.items():
        row = _row_at(data_by_symbol[symbol], date)
        if row is None:
            continue
        previous_close = _previous_close(data_by_symbol[symbol], date)
        current_shares = positions.get(symbol).shares if symbol in positions else 0
        current_value = current_shares * float(row["Close"])
        buy_value = target_value - current_value
        if buy_value <= float(row["Close"]):
            continue

        allowed, reason = can_buy(row.to_dict(), previous_close, request.trading)
        if not allowed:
            skipped_trades.append({"symbol": symbol, "side": "buy", "reason": reason})
            continue

        price = apply_slippage(float(row["Close"]), "buy", request.trading.slippage_pct)
        shares = round_lot_shares(min(buy_value, cash) / price, request.trading.lot_size)
        while shares > 0:
            amount = shares * price
            cost = calculate_trade_cost(amount, "buy", request.trading)
            if amount + cost <= cash:
                break
            shares -= request.trading.lot_size
        if shares <= 0:
            skipped_trades.append({"symbol": symbol, "side": "buy", "reason": "insufficient_cash"})
            continue

        amount = shares * price
        cost = calculate_trade_cost(amount, "buy", request.trading)
        cash -= amount + cost
        if symbol in positions:
            positions[symbol].shares += shares
        else:
            positions[symbol] = Position(symbol=symbol, shares=shares, entry_date=_date_str(date))
        trade = _trade_record(date, symbol, "buy", shares, price, amount, cost, "rebalance")
        trades.append(trade)
        day_trades.append(trade)
    return cash


def _trade_record(
    date: pd.Timestamp,
    symbol: str,
    side: str,
    shares: int,
    price: float,
    amount: float,
    cost: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "date": _date_str(date),
        "symbol": symbol,
        "side": side,
        "shares": int(shares),
        "price": round(price, 6),
        "amount": round(amount, 6),
        "cost": round(cost, 6),
        "reason": reason,
    }


def _portfolio_value(
    date: pd.Timestamp,
    cash: float,
    positions: dict[str, Position],
    data_by_symbol: dict[str, pd.DataFrame],
) -> float:
    return cash + _gross_exposure(date, positions, data_by_symbol)


def _gross_exposure(
    date: pd.Timestamp,
    positions: dict[str, Position],
    data_by_symbol: dict[str, pd.DataFrame],
) -> float:
    total = 0.0
    for symbol, position in positions.items():
        row = _row_at(data_by_symbol[symbol], date)
        if row is not None:
            total += position.shares * float(row["Close"])
    return total


def _row_at(data: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    if date in data.index:
        return data.loc[date]
    previous = data[data.index <= date]
    if previous.empty:
        return None
    return previous.iloc[-1]


def _previous_close(data: pd.DataFrame, date: pd.Timestamp) -> float:
    previous = data[data.index < date]
    if previous.empty:
        row = _row_at(data, date)
        return float(row["Close"]) if row is not None else 0.0
    return float(previous.iloc[-1]["Close"])


def _ranking_rows(date: pd.Timestamp, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "date": _date_str(date)} for row in rows]


def _positions_payload(
    date: pd.Timestamp,
    positions: dict[str, Position],
    data_by_symbol: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    payload = []
    for symbol, position in positions.items():
        row = _row_at(data_by_symbol[symbol], date)
        if row is None:
            continue
        price = float(row["Close"])
        payload.append({
            "date": _date_str(date),
            "symbol": symbol,
            "shares": position.shares,
            "price": round(price, 6),
            "market_value": round(position.shares * price, 6),
            "holding_bars": position.holding_bars,
        })
    return payload


def _fill_drawdowns(equity_curve: list[dict[str, Any]]) -> None:
    peak = 0.0
    for point in equity_curve:
        equity = float(point["equity"])
        peak = max(peak, equity)
        point["drawdown_pct"] = round((equity / peak - 1) * 100, 6) if peak else 0.0


def _summary(
    equity_curve: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    rebalance_events: list[dict[str, Any]],
    initial_cash: float,
) -> dict[str, Any]:
    if not equity_curve:
        return {
            "final_equity": initial_cash,
            "total_return_pct": 0.0,
            "annual_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "turnover": 0.0,
            "rebalances": 0,
            "trades": 0,
            "final_gross_exposure": 0.0,
        }

    final_equity = float(equity_curve[-1]["equity"])
    total_return_pct = (final_equity / initial_cash - 1) * 100
    dates = [pd.Timestamp(point["date"]) for point in equity_curve]
    years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
    annual_return_pct = ((final_equity / initial_cash) ** (1 / years) - 1) * 100 if initial_cash else 0.0
    max_drawdown_pct = abs(min(float(point["drawdown_pct"]) for point in equity_curve))
    equity_series = pd.Series([float(point["equity"]) for point in equity_curve])
    returns = equity_series.pct_change().dropna()
    sharpe = 0.0
    if not returns.empty and returns.std() not in (0, math.nan) and not math.isnan(float(returns.std())):
        sharpe = float(returns.mean() / returns.std() * math.sqrt(252))
    turnover = sum(float(trade["amount"]) for trade in trades) / initial_cash if initial_cash else 0.0

    return {
        "final_equity": round(final_equity, 6),
        "total_return_pct": round(total_return_pct, 6),
        "annual_return_pct": round(annual_return_pct, 6),
        "max_drawdown_pct": round(max_drawdown_pct, 6),
        "sharpe": round(sharpe, 6),
        "turnover": round(turnover, 6),
        "rebalances": len(rebalance_events),
        "trades": len(trades),
        "final_gross_exposure": round(float(equity_curve[-1].get("gross_exposure", 0)), 6),
    }


def _risk_flags(
    summary: dict[str, Any],
    warnings: list[str],
    rebalance_events: list[dict[str, Any]],
) -> list[str]:
    flags = []
    if summary.get("rebalances", 0) < 2:
        flags.append("too_few_rebalances")
    if summary.get("max_drawdown_pct", 0) > 30:
        flags.append("high_drawdown")
    if summary.get("turnover", 0) > 10:
        flags.append("high_turnover")
    if warnings:
        flags.append("data_gaps")
    if summary.get("final_gross_exposure", 0) < 0.2:
        flags.append("underinvested")
    if any(len(event.get("selected_symbols", [])) == 0 for event in rebalance_events):
        flags.append("too_few_selected")
    return flags


def _date_str(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")
