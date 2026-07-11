from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from a_share_rules import apply_slippage, calculate_trade_cost, can_buy, can_sell, round_lot_shares
from selection_engine import build_trading_calendar
from signal_portfolio_models import SignalPortfolioBacktestRequest, SignalPortfolioBacktestResult
from universe_scan_runner import load_universe_scan_data


@dataclass
class SignalPosition:
    symbol: str
    shares: int
    entry_date: str
    entry_price: float
    entry_cost: float
    stop_price: float
    target_price: float
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
                    "trigger_price": round(float(row["High"]), 6),
                    "pin_low": round(float(row["Low"]), 6),
                    "atr": round(float(row["atr"]), 6),
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
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    volume = frame["Volume"].astype(float)
    frame["ma_short"] = close.rolling(config.short_ma_period).mean()
    frame["ma_medium"] = close.rolling(config.medium_ma_period).mean()
    frame["ma_long"] = close.rolling(config.long_ma_period).mean()
    frame["support"] = low.shift(1).rolling(config.support_lookback).min()
    frame["average_volume"] = volume.shift(1).rolling(config.volume_lookback).mean()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr"] = true_range.rolling(config.atr_period).mean()
    signals = []
    strengths = []
    min_bars = max(
        config.long_ma_period,
        config.support_lookback + 1,
        config.volume_lookback + 1,
        config.atr_period + 1,
    )
    for index in range(len(frame)):
        if index + 1 < min_bars:
            signals.append(False)
            strengths.append(0.0)
            continue
        current = frame.iloc[index]
        signal = _is_trend_pullback_pin_bar(current, config)
        signals.append(signal)
        strengths.append(_pin_bar_signal_strength(current) if signal else 0.0)
    frame["entry_signal"] = signals
    frame["signal_strength"] = strengths
    return frame


def _is_trend_pullback_pin_bar(current: pd.Series, config) -> bool:
    required_values = [
        current["Open"],
        current["High"],
        current["Low"],
        current["Close"],
        current["Volume"],
        current["ma_short"],
        current["ma_medium"],
        current["ma_long"],
        current["support"],
        current["average_volume"],
        current["atr"],
    ]
    if not all(math.isfinite(float(value)) for value in required_values):
        return False

    open_price = float(current["Open"])
    high = float(current["High"])
    low = float(current["Low"])
    close = float(current["Close"])
    ma_short = float(current["ma_short"])
    ma_medium = float(current["ma_medium"])
    ma_long = float(current["ma_long"])
    support = float(current["support"])
    candle_range = high - low
    if candle_range <= 0 or ma_short <= 0 or support <= 0:
        return False

    trend_ok = ma_short > ma_medium > ma_long and close > ma_medium
    near_ma = abs(close / ma_short - 1) * 100 <= config.ma_distance_pct
    near_support = abs(low / support - 1) * 100 <= config.support_tolerance_pct

    body = abs(close - open_price)
    lower_shadow = min(open_price, close) - low
    upper_shadow = high - max(open_price, close)
    close_location_pct = (close - low) / candle_range * 100
    pin_bar_ok = (
        lower_shadow >= body * config.lower_shadow_body_ratio
        and body / candle_range * 100 <= config.max_body_range_pct
        and close_location_pct >= config.min_close_location_pct
        and upper_shadow / candle_range * 100 <= config.max_upper_shadow_range_pct
    )
    volume_ok = (
        float(current["average_volume"]) > 0
        and float(current["Volume"])
        >= float(current["average_volume"]) * config.volume_multiplier
    )
    return trend_ok and (near_ma or near_support) and pin_bar_ok and volume_ok


def _pin_bar_signal_strength(current: pd.Series) -> float:
    candle_range = float(current["High"]) - float(current["Low"])
    volume_ratio = float(current["Volume"]) / float(current["average_volume"])
    close_location = (float(current["Close"]) - float(current["Low"])) / candle_range
    trend_spread = (
        float(current["ma_short"]) / float(current["ma_medium"]) - 1
        + float(current["ma_medium"]) / float(current["ma_long"]) - 1
    )
    return volume_ratio * 0.5 + close_location * 0.3 + trend_spread * 100 * 0.2


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
        elif float(row["High"]) >= position.target_price:
            reason = "reward_target"
            raw_price = max(float(row["Open"]), position.target_price)
        if reason is None:
            ma_short = float(row.get("ma_short", math.nan))
            if math.isfinite(ma_short) and float(row["Close"]) < ma_short:
                position.exit_next_open_reason = "trend_weak"
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
        trigger_price = float(candidate["trigger_price"])
        open_price = float(row["Open"])
        if open_price > trigger_price * (1 + request.strategy.max_entry_gap_pct / 100):
            continue
        if float(row["High"]) < trigger_price:
            continue
        raw_entry_price = max(open_price, trigger_price)
        execution_row = {**row.to_dict(), "Close": raw_entry_price}
        allowed, _ = can_buy(execution_row, _previous_close(data, date), request.trading)
        if not allowed:
            continue

        equity = _portfolio_value(date, cash, positions, data_by_symbol)
        price = apply_slippage(raw_entry_price, "buy", request.trading.slippage_pct)
        structural_stop = float(candidate["pin_low"]) - request.strategy.price_tick
        atr_stop = price - float(candidate["atr"])
        stop_price = min(structural_stop, atr_stop)
        risk_per_share = price - stop_price
        if price <= 0 or risk_per_share <= 0:
            continue
        stop_distance_pct = risk_per_share / price * 100
        if not (
            request.strategy.min_stop_distance_pct
            <= stop_distance_pct
            <= request.strategy.max_stop_distance_pct
        ):
            continue

        slot_pct = min(request.risk.max_position_pct, request.risk.target_gross_exposure / request.risk.max_positions)
        position_budget = min(cash, equity * slot_pct)
        risk_budget = equity * request.strategy.risk_per_trade_pct / 100
        budget_shares = round_lot_shares(position_budget / price, request.trading.lot_size)
        risk_shares = round_lot_shares(risk_budget / risk_per_share, request.trading.lot_size)
        shares = min(budget_shares, risk_shares)
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
        target_price = price + risk_per_share * request.strategy.reward_risk_ratio
        exit_next_open_reason = None
        if float(row["Low"]) <= stop_price:
            exit_next_open_reason = "stop_loss_t1"
        elif float(row["High"]) >= target_price:
            exit_next_open_reason = "reward_target_t1"
        elif float(row["Close"]) < float(row["ma_short"]):
            exit_next_open_reason = "trend_weak"
        cash -= amount + cost
        positions[symbol] = SignalPosition(
            symbol=symbol,
            shares=shares,
            entry_date=_date_str(date),
            entry_price=price,
            entry_cost=cost,
            stop_price=stop_price,
            target_price=target_price,
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
        rows.append({"symbol": symbol, "shares": position.shares, "entry_date": position.entry_date, "entry_price": position.entry_price, "stop_price": round(position.stop_price, 6), "target_price": round(position.target_price, 6), "price": round(price, 6), "market_value": round(position.shares * price, 6), "unrealized_pnl": round(position.shares * (price - position.entry_price) - position.entry_cost, 6), "holding_bars": position.holding_bars})
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
