from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from a_share_rules import apply_slippage, calculate_trade_cost, can_buy, can_sell, round_lot_shares
from selection_engine import build_trading_calendar
from signal_portfolio_models import SignalPortfolioBacktestRequest, SignalPortfolioBacktestResult
from strategies.boll_macd_breakout import bollinger_middle, bollinger_upper
from universe_scan_runner import load_universe_scan_data


@dataclass
class SignalPosition:
    symbol: str
    shares: int
    entry_date: str
    entry_price: float
    entry_cost: float
    stop_price: float
    holding_bars: int = 0
    exit_next_open_reason: str | None = None


def run_signal_portfolio_backtest(
    request: SignalPortfolioBacktestRequest,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SignalPortfolioBacktestResult:
    scan = load_universe_scan_data(request, progress_callback=progress_callback)
    return run_signal_portfolio_with_data(
        request,
        scan.data_by_symbol,
        providers=scan.providers,
        warnings=scan.warnings,
        diagnostics=scan.diagnostics,
        progress_callback=progress_callback,
    )


def run_signal_portfolio_with_data(
    request: SignalPortfolioBacktestRequest,
    data_by_symbol: dict[str, pd.DataFrame],
    *,
    providers: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SignalPortfolioBacktestResult:
    if not data_by_symbol:
        raise ValueError("多股票信号组合没有可用行情数据")
    _emit(progress_callback, phase="building_signals", screened_count=len(data_by_symbol))
    signal_frames = {
        symbol: _build_signal_frame(data, request)
        for symbol, data in data_by_symbol.items()
    }
    calendar = build_trading_calendar(signal_frames)
    calendar = [
        date for date in calendar
        if pd.Timestamp(request.start_date) <= pd.Timestamp(date) <= pd.Timestamp(request.end_date)
    ]
    if not calendar:
        raise ValueError("回测区间内没有交易日")

    cash = float(request.initial_cash)
    positions: dict[str, SignalPosition] = {}
    pending_entries: dict[str, dict[str, Any]] = {}
    equity_curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    signal_events: list[dict[str, Any]] = []
    contributions: dict[str, dict[str, float | int]] = {}
    peak_equity = float(request.initial_cash)
    entry_blocked = False

    for day_index, date in enumerate(calendar):
        _emit(
            progress_callback,
            phase="signal_backtesting",
            completed_days=day_index,
            total_days=len(calendar),
            current_date=_date_str(date),
        )
        for symbol, position in positions.items():
            if date in signal_frames[symbol].index and _date_str(date) != position.entry_date:
                position.holding_bars += 1

        cash = _execute_exits(date, cash, positions, signal_frames, request, trades, contributions)
        equity_before_entries = _portfolio_value(date, cash, positions, signal_frames)
        peak_equity = max(peak_equity, equity_before_entries)
        drawdown_pct = (equity_before_entries / peak_equity - 1) * 100 if peak_equity else 0.0
        if request.risk.max_drawdown_stop_pct is not None and abs(drawdown_pct) >= request.risk.max_drawdown_stop_pct:
            entry_blocked = True

        cash = _execute_pending_entries(
            date,
            cash,
            positions,
            pending_entries,
            signal_frames,
            request,
            trades,
            contributions,
            entry_blocked=entry_blocked,
        )

        for symbol, frame in signal_frames.items():
            if symbol in positions or symbol in pending_entries or date not in frame.index:
                continue
            row = frame.loc[date]
            if bool(row.get("entry_signal", False)):
                event = {
                    "date": _date_str(date),
                    "symbol": symbol,
                    "strength": round(float(row.get("signal_strength", 0)), 8),
                }
                pending_entries[symbol] = event
                signal_events.append(event)

        equity = _portfolio_value(date, cash, positions, signal_frames)
        peak_equity = max(peak_equity, equity)
        gross = _gross_exposure(date, positions, signal_frames)
        equity_curve.append({
            "date": _date_str(date),
            "equity": round(equity, 6),
            "cash": round(cash, 6),
            "gross_exposure": round(gross / equity, 6) if equity else 0.0,
            "drawdown_pct": round((equity / peak_equity - 1) * 100, 6) if peak_equity else 0.0,
        })

    summary = _summary(equity_curve, trades, request.initial_cash)
    result_diagnostics = dict(diagnostics or {})
    result_diagnostics.update({
        "loaded_symbols": len(data_by_symbol),
        "signal_count": len(signal_events),
        "traded_symbols": len({trade["symbol"] for trade in trades}),
        "providers": providers or {},
    })
    return SignalPortfolioBacktestResult(
        summary=summary,
        equity_curve=equity_curve,
        positions=_position_rows(calendar[-1], positions, signal_frames),
        trades=trades,
        symbol_contributions=_contribution_rows(contributions, positions, calendar[-1], signal_frames),
        signal_events=signal_events,
        data_warnings=list(warnings or []),
        risk_flags=_risk_flags(summary, warnings or [], entry_blocked),
        scan_diagnostics=result_diagnostics,
        config=request.model_dump(mode="json"),
    )


def _build_signal_frame(data: pd.DataFrame, request: SignalPortfolioBacktestRequest) -> pd.DataFrame:
    config = request.strategy
    frame = data.copy()
    close = frame["Close"].astype(float)
    frame["middle"] = bollinger_middle(close, config.boll_period)
    frame["upper"] = bollinger_upper(close, config.boll_period, config.boll_stddev)
    frame["lower"] = _bollinger_lower(close, config.boll_period, config.boll_stddev)
    signals = []
    strengths = []
    min_bars = config.boll_period + config.confirmation_days
    for index in range(len(frame)):
        if index < 2 or index + 1 < min_bars:
            signals.append(False)
            strengths.append(0.0)
            continue
        before_cross = frame.iloc[index - 2]
        cross_day = frame.iloc[index - 1]
        confirmation_day = frame.iloc[index]
        signal = _is_two_day_middle_recovery_entry(
            before_cross,
            cross_day,
            confirmation_day,
        )
        signals.append(signal)
        close_price = float(confirmation_day["Close"])
        upper = float(confirmation_day["upper"])
        upside_room = (upper - close_price) / close_price if close_price > 0 else 0.0
        strengths.append(upside_room if signal else 0.0)
    frame["entry_signal"] = signals
    frame["signal_strength"] = strengths
    return frame


def _bollinger_lower(values, period: int, stddev: float):
    prices = pd.Series(values, dtype="float64")
    rolling = prices.rolling(period)
    return (rolling.mean() - rolling.std(ddof=0) * stddev).to_numpy()


def _is_two_day_middle_recovery_entry(
    before_cross: pd.Series,
    cross_day: pd.Series,
    confirmation_day: pd.Series,
) -> bool:
    required_values = [
        before_cross["Close"],
        before_cross["middle"],
        cross_day["Close"],
        cross_day["middle"],
        cross_day["Low"],
        cross_day["lower"],
        cross_day["High"],
        cross_day["upper"],
        confirmation_day["Close"],
        confirmation_day["middle"],
        confirmation_day["Low"],
        confirmation_day["lower"],
        confirmation_day["High"],
        confirmation_day["upper"],
    ]
    if not all(math.isfinite(float(value)) for value in required_values):
        return False

    crossed_middle = (
        float(before_cross["Close"]) <= float(before_cross["middle"])
        and float(cross_day["Close"]) > float(cross_day["middle"])
    )
    cross_day_is_stable = (
        float(cross_day["Low"]) >= float(cross_day["lower"])
        and float(cross_day["High"]) < float(cross_day["upper"])
    )
    confirmation_day_is_stable = (
        float(confirmation_day["Close"]) > float(confirmation_day["middle"])
        and float(confirmation_day["Low"]) >= float(confirmation_day["lower"])
        and float(confirmation_day["High"]) < float(confirmation_day["upper"])
    )
    return crossed_middle and cross_day_is_stable and confirmation_day_is_stable


def _execute_exits(date, cash, positions, data_by_symbol, request, trades, contributions):
    for symbol in list(positions):
        position = positions[symbol]
        data = data_by_symbol[symbol]
        if date not in data.index or position.holding_bars < 1:
            continue
        row = data.loc[date]
        reason = None
        raw_price = None
        if position.exit_next_open_reason:
            reason = position.exit_next_open_reason
            raw_price = float(row["Open"])
        elif float(row["Low"]) <= position.stop_price:
            reason = "stop_loss"
            raw_price = min(float(row["Open"]), position.stop_price)
        else:
            upper_price = _previous_indicator(data, date, "upper")
            if upper_price is not None and float(row["High"]) >= upper_price:
                reason = "upper_band"
                raw_price = max(float(row["Open"]), upper_price)
        if reason is None:
            continue
        allowed, _ = can_sell(row.to_dict(), _previous_close(data, date), position.holding_bars, request.trading)
        if not allowed:
            continue
        price = apply_slippage(raw_price, "sell", request.trading.slippage_pct)
        amount = position.shares * price
        cost = calculate_trade_cost(amount, "sell", request.trading)
        pnl = amount - cost - position.shares * position.entry_price - position.entry_cost
        cash += amount - cost
        trades.append(_trade(date, symbol, "sell", position.shares, price, amount, cost, reason, pnl))
        bucket = contributions.setdefault(symbol, {"realized_pnl": 0.0, "round_trips": 0})
        bucket["realized_pnl"] = float(bucket["realized_pnl"]) + pnl
        bucket["round_trips"] = int(bucket["round_trips"]) + 1
        positions.pop(symbol)
    return cash


def _execute_pending_entries(date, cash, positions, pending, data_by_symbol, request, trades, contributions, *, entry_blocked):
    candidates = sorted(pending.values(), key=lambda row: float(row["strength"]), reverse=True)
    for candidate in candidates:
        symbol = candidate["symbol"]
        data = data_by_symbol[symbol]
        if pd.Timestamp(date) <= pd.Timestamp(candidate["date"]) or date not in data.index:
            continue
        pending.pop(symbol, None)
        if entry_blocked or len(positions) >= request.risk.max_positions:
            continue
        row = data.loc[date]
        previous = data[data.index < date]
        previous_volume = float(previous.iloc[-1]["Volume"]) if not previous.empty else float(row["Volume"])
        open_row = {**row.to_dict(), "Close": float(row["Open"]), "Volume": previous_volume}
        allowed, _ = can_buy(open_row, _previous_close(data, date), request.trading)
        if not allowed:
            continue
        equity = _portfolio_value(date, cash, positions, data_by_symbol)
        slot_pct = min(request.risk.max_position_pct, request.risk.target_gross_exposure / request.risk.max_positions)
        budget = min(cash, equity * slot_pct)
        price = apply_slippage(float(row["Open"]), "buy", request.trading.slippage_pct)
        shares = round_lot_shares(budget / price, request.trading.lot_size)
        while shares > 0:
            amount = shares * price
            cost = calculate_trade_cost(amount, "buy", request.trading)
            if amount + cost <= cash:
                break
            shares -= request.trading.lot_size
        if shares <= 0:
            continue
        amount = shares * price
        cost = calculate_trade_cost(amount, "buy", request.trading)
        stop_price = price * (1 - request.strategy.stop_loss_pct / 100)
        previous_upper = float(previous.iloc[-1]["upper"]) if not previous.empty else None
        exit_next_open_reason = None
        if float(row["Low"]) <= stop_price:
            exit_next_open_reason = "stop_loss_t1"
        elif (
            previous_upper is not None
            and math.isfinite(previous_upper)
            and float(row["High"]) >= previous_upper
        ):
            exit_next_open_reason = "upper_band_t1"
        cash -= amount + cost
        positions[symbol] = SignalPosition(
            symbol=symbol,
            shares=shares,
            entry_date=_date_str(date),
            entry_price=price,
            entry_cost=cost,
            stop_price=stop_price,
            exit_next_open_reason=exit_next_open_reason,
        )
        contributions.setdefault(symbol, {"realized_pnl": 0.0, "round_trips": 0})
        trades.append(_trade(date, symbol, "buy", shares, price, amount, cost, "signal", None))
    return cash


def _trade(date, symbol, side, shares, price, amount, cost, reason, pnl):
    return {"date": _date_str(date), "symbol": symbol, "side": side, "shares": int(shares), "price": round(price, 6), "amount": round(amount, 6), "cost": round(cost, 6), "reason": reason, "pnl": None if pnl is None else round(pnl, 6)}


def _row_at(data, date):
    if date in data.index:
        return data.loc[date]
    previous = data[data.index <= date]
    return None if previous.empty else previous.iloc[-1]


def _previous_close(data, date):
    previous = data[data.index < date]
    return float(previous.iloc[-1]["Close"]) if not previous.empty else float(data.loc[date]["Open"])


def _previous_indicator(data, date, column):
    previous = data[data.index < date]
    if previous.empty or column not in previous.columns:
        return None
    value = float(previous.iloc[-1][column])
    return value if math.isfinite(value) else None


def _gross_exposure(date, positions, data_by_symbol):
    return sum(position.shares * float(_row_at(data_by_symbol[symbol], date)["Close"]) for symbol, position in positions.items() if _row_at(data_by_symbol[symbol], date) is not None)


def _portfolio_value(date, cash, positions, data_by_symbol):
    return cash + _gross_exposure(date, positions, data_by_symbol)


def _summary(curve, trades, initial_cash):
    final = float(curve[-1]["equity"]) if curve else float(initial_cash)
    total_return = (final / initial_cash - 1) * 100
    years = max((pd.Timestamp(curve[-1]["date"]) - pd.Timestamp(curve[0]["date"])).days / 365.25, 1 / 365.25) if curve else 1
    annual = ((final / initial_cash) ** (1 / years) - 1) * 100 if initial_cash and final > 0 else -100.0
    returns = pd.Series([point["equity"] for point in curve], dtype="float64").pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * math.sqrt(252)) if len(returns) > 1 and float(returns.std()) > 0 else 0.0
    sells = [trade for trade in trades if trade["side"] == "sell"]
    wins = [trade for trade in sells if float(trade.get("pnl") or 0) > 0]
    return {"final_equity": round(final, 6), "total_return_pct": round(total_return, 6), "annual_return_pct": round(annual, 6), "max_drawdown_pct": round(abs(min((point["drawdown_pct"] for point in curve), default=0)), 6), "sharpe": round(sharpe, 6), "trades": len(sells), "orders": len(trades), "win_rate_pct": round(len(wins) / len(sells) * 100, 6) if sells else 0.0, "turnover": round(sum(trade["amount"] for trade in trades) / initial_cash, 6) if initial_cash else 0.0, "final_gross_exposure": curve[-1]["gross_exposure"] if curve else 0.0}


def _position_rows(date, positions, data_by_symbol):
    rows = []
    for symbol, position in positions.items():
        price = float(_row_at(data_by_symbol[symbol], date)["Close"])
        rows.append({"symbol": symbol, "shares": position.shares, "entry_date": position.entry_date, "entry_price": position.entry_price, "price": round(price, 6), "market_value": round(position.shares * price, 6), "unrealized_pnl": round(position.shares * (price - position.entry_price) - position.entry_cost, 6), "holding_bars": position.holding_bars})
    return rows


def _contribution_rows(contributions, positions, date, data_by_symbol):
    rows = []
    for symbol in sorted(contributions):
        bucket = contributions[symbol]
        unrealized = 0.0
        if symbol in positions:
            position = positions[symbol]
            price = float(_row_at(data_by_symbol[symbol], date)["Close"])
            unrealized = position.shares * (price - position.entry_price) - position.entry_cost
        realized = float(bucket["realized_pnl"])
        rows.append({"symbol": symbol, "round_trips": int(bucket["round_trips"]), "realized_pnl": round(realized, 6), "unrealized_pnl": round(unrealized, 6), "total_pnl": round(realized + unrealized, 6)})
    return sorted(rows, key=lambda row: row["total_pnl"], reverse=True)


def _risk_flags(summary, warnings, entry_blocked):
    flags = []
    if summary["trades"] < 5:
        flags.append("too_few_trades")
    if summary["max_drawdown_pct"] > 30:
        flags.append("high_drawdown")
    if summary["final_gross_exposure"] < 0.2:
        flags.append("underinvested")
    if warnings:
        flags.append("data_gaps")
    if entry_blocked:
        flags.append("drawdown_entry_stop")
    return flags


def _emit(callback, **event):
    if callback is not None:
        callback(event)


def _date_str(date):
    return pd.Timestamp(date).strftime("%Y-%m-%d")
