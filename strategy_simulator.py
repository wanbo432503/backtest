from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from a_share_rules import (
    apply_slippage,
    calculate_trade_cost,
    can_buy,
    can_sell,
    round_lot_shares,
)
from optimization_models import AShareTradingConfig
from selection_engine import build_trading_calendar
from strategy_engine import (
    EntryIntent,
    RiskIntent,
    SimulationPosition,
    SimulationResult,
    StrategyBarContext,
    StrategyDefinition,
)


@dataclass(frozen=True)
class SimulationConfig:
    initial_cash: float
    max_positions: int = 1
    max_position_pct: float = 1.0
    target_gross_exposure: float = 1.0
    max_drawdown_stop_pct: float | None = None
    trading: AShareTradingConfig = field(default_factory=AShareTradingConfig)
    start_date: str | None = None
    end_date: str | None = None
    min_entry_history_bars: int = 0
    entry_history_start_date: str | None = None


@dataclass
class _Position:
    symbol: str
    shares: int
    entry_date: str
    entry_price: float
    entry_cost: float
    holding_bars: int = 0
    highest_price: float | None = None
    risk: RiskIntent | None = None

    def public(self) -> SimulationPosition:
        return SimulationPosition(
            symbol=self.symbol,
            shares=self.shares,
            entry_date=self.entry_date,
            entry_price=self.entry_price,
            entry_cost=self.entry_cost,
            holding_bars=self.holding_bars,
            highest_price=self.highest_price,
            risk=self.risk,
        )


@dataclass
class _PendingEntry:
    symbol: str
    created_date: pd.Timestamp
    intent: EntryIntent
    risk_multiplier: float = 1.0
    attempts: int = 0


EntryRiskMultiplier = Callable[[str, pd.Timestamp, pd.Series, EntryIntent], float]


def run_strategy_simulation(
    definition: StrategyDefinition,
    strategy_config,
    data_by_symbol: dict[str, pd.DataFrame],
    simulation_config: SimulationConfig,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    entry_risk_multiplier: EntryRiskMultiplier | None = None,
) -> SimulationResult:
    if not data_by_symbol:
        raise ValueError("strategy simulation requires at least one symbol")
    frames = {}
    for symbol, data in data_by_symbol.items():
        try:
            frames[symbol] = definition.prepare_frame(
                _normalized_frame(data),
                strategy_config,
            )
        except Exception as exc:
            raise RuntimeError(
                f"strategy '{definition.strategy_id}' symbol {symbol} preparation failed: {exc}"
            ) from exc
    calendar = build_trading_calendar(frames)
    if simulation_config.start_date:
        calendar = [date for date in calendar if date >= pd.Timestamp(simulation_config.start_date)]
    if simulation_config.end_date:
        calendar = [date for date in calendar if date <= pd.Timestamp(simulation_config.end_date)]
    if not calendar:
        raise ValueError("strategy simulation has no trading dates")

    cash = float(simulation_config.initial_cash)
    peak_equity = cash
    entry_blocked = False
    positions: dict[str, _Position] = {}
    pending_entries: dict[str, _PendingEntry] = {}
    pending_exits: dict[str, str] = {}
    last_exit_day_index: dict[str, int] = {}
    strategy_states: dict[str, dict[str, Any]] = {
        symbol: {} for symbol in frames
    }
    trades: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    contributions: dict[str, dict[str, float | int]] = {}
    diagnostics = {
        "expired_entry_count": 0,
        "rejected_entry_count": 0,
        "insufficient_entry_history_count": 0,
        "drawdown_entry_stop": False,
    }
    entry_history_start_date = (
        pd.Timestamp(simulation_config.entry_history_start_date)
        if simulation_config.entry_history_start_date
        else None
    )

    for day_index, date in enumerate(calendar):
        _emit(
            progress_callback,
            phase="backtesting",
            completed_days=day_index,
            total_days=len(calendar),
            current_date=_date_str(date),
        )
        for symbol, position in positions.items():
            if date in frames[symbol].index and _date_str(date) != position.entry_date:
                position.holding_bars += 1
                position.highest_price = max(
                    position.highest_price or position.entry_price,
                    float(frames[symbol].loc[date, "Close"]),
                )

        for symbol in list(positions):
            if date not in frames[symbol].index:
                continue
            position = positions[symbol]
            row = frames[symbol].loc[date]
            reason, raw_price = _exit_for_bar(
                row,
                position,
                pending_exits.get(symbol),
            )
            if reason is None or raw_price is None:
                continue
            execution_row = {**row.to_dict(), "Close": raw_price}
            allowed, _ = can_sell(
                execution_row,
                _previous_close(frames[symbol], date),
                position.holding_bars,
                simulation_config.trading,
            )
            if not allowed:
                continue
            price = apply_slippage(raw_price, "sell", simulation_config.trading.slippage_pct)
            amount = position.shares * price
            cost = calculate_trade_cost(amount, "sell", simulation_config.trading)
            pnl = amount - cost - position.shares * position.entry_price - position.entry_cost
            cash += amount - cost
            trades.append(
                _trade(date, symbol, "sell", position.shares, price, amount, cost, reason, pnl)
            )
            bucket = contributions.setdefault(symbol, {"realized_pnl": 0.0, "round_trips": 0})
            bucket["realized_pnl"] = float(bucket["realized_pnl"]) + pnl
            bucket["round_trips"] = int(bucket["round_trips"]) + 1
            positions.pop(symbol)
            pending_exits.pop(symbol, None)
            last_exit_day_index[symbol] = day_index

        equity_before_entries = _portfolio_value_at_execution(
            date,
            cash,
            positions,
            frames,
        )
        peak_equity = max(peak_equity, equity_before_entries)
        drawdown_pct = (
            (equity_before_entries / peak_equity - 1) * 100
            if peak_equity
            else 0.0
        )
        if (
            simulation_config.max_drawdown_stop_pct is not None
            and abs(drawdown_pct) >= simulation_config.max_drawdown_stop_pct
        ):
            entry_blocked = True
            diagnostics["drawdown_entry_stop"] = True

        candidates = sorted(
            pending_entries.values(),
            key=lambda pending: (-float(pending.intent.strength), pending.symbol),
        )
        for pending in candidates:
            symbol = pending.symbol
            frame = frames[symbol]
            if date <= pending.created_date or date not in frame.index:
                continue
            if entry_blocked or symbol in positions or len(positions) >= simulation_config.max_positions:
                diagnostics["rejected_entry_count"] += 1
                pending_entries.pop(symbol, None)
                continue
            row = frame.loc[date]
            raw_price = _entry_price(pending.intent, row)
            pending.attempts += 1
            if raw_price is None:
                if pending.attempts >= pending.intent.expires_after_bars:
                    diagnostics["expired_entry_count"] += 1
                    pending_entries.pop(symbol, None)
                continue
            max_gap_pct = pending.intent.metadata.get("max_entry_gap_pct")
            if (
                pending.intent.trigger_price is not None
                and max_gap_pct is not None
                and float(row["Open"])
                > pending.intent.trigger_price * (1 + float(max_gap_pct) / 100)
            ):
                diagnostics["rejected_entry_count"] += 1
                pending_entries.pop(symbol, None)
                continue
            execution_row = {**row.to_dict(), "Close": raw_price}
            allowed, _ = can_buy(
                execution_row,
                _previous_close(frame, date),
                simulation_config.trading,
            )
            if not allowed:
                diagnostics["rejected_entry_count"] += 1
                pending_entries.pop(symbol, None)
                continue

            equity = _portfolio_value_at_execution(date, cash, positions, frames)
            gross = _gross_exposure_at_execution(date, positions, frames)
            multiplier = pending.risk_multiplier
            price = apply_slippage(raw_price, "buy", simulation_config.trading.slippage_pct)
            suggested_budget = equity * pending.intent.suggested_position_pct * multiplier
            position_cap = equity * simulation_config.max_position_pct
            exposure_cap = max(0.0, equity * simulation_config.target_gross_exposure - gross)
            position_budget = min(cash, suggested_budget, position_cap, exposure_cap)
            budget_shares = round_lot_shares(
                position_budget / price,
                simulation_config.trading.lot_size,
            )
            risk = _risk_for_fill(pending.intent, price)
            risk_shares = budget_shares
            if (
                risk is not None
                and risk.risk_budget_pct is not None
                and risk.risk_per_share is not None
                and risk.risk_per_share > 0
            ):
                risk_shares = round_lot_shares(
                    equity * risk.risk_budget_pct * multiplier / risk.risk_per_share,
                    simulation_config.trading.lot_size,
                )
            shares = min(budget_shares, risk_shares)
            while shares > 0:
                amount = shares * price
                cost = calculate_trade_cost(amount, "buy", simulation_config.trading)
                if amount + cost <= cash:
                    break
                shares -= simulation_config.trading.lot_size
            if shares <= 0:
                diagnostics["rejected_entry_count"] += 1
                pending_entries.pop(symbol, None)
                continue
            amount = shares * price
            cost = calculate_trade_cost(amount, "buy", simulation_config.trading)
            cash -= amount + cost
            positions[symbol] = _Position(
                symbol=symbol,
                shares=shares,
                entry_date=_date_str(date),
                entry_price=price,
                entry_cost=cost,
                highest_price=price,
                risk=risk,
            )
            contributions.setdefault(symbol, {"realized_pnl": 0.0, "round_trips": 0})
            trades.append(_trade(date, symbol, "buy", shares, price, amount, cost, "signal", None))
            pending_entries.pop(symbol, None)

        for symbol, frame in frames.items():
            if date not in frame.index or symbol in pending_exits:
                continue
            location = frame.index.get_loc(date)
            if not isinstance(location, int):
                raise ValueError(f"duplicate trading date for {symbol}: {_date_str(date)}")
            context = StrategyBarContext(
                symbol=symbol,
                frame=frame,
                bar_index=location,
                config=strategy_config,
                position=positions[symbol].public() if symbol in positions else None,
                state=strategy_states[symbol],
                bars_since_exit=(
                    day_index - last_exit_day_index[symbol]
                    if symbol in last_exit_day_index
                    else None
                ),
                entry_history_start_date=entry_history_start_date,
            )
            try:
                decision = definition.evaluate(context)
            except Exception as exc:
                raise RuntimeError(
                    f"strategy '{definition.strategy_id}' symbol {symbol} "
                    f"date {_date_str(date)} evaluation failed: {exc}"
                ) from exc
            if decision.next_state is not None:
                strategy_states[symbol] = dict(decision.next_state)
            if decision.risk_update is not None and symbol in positions:
                positions[symbol].risk = decision.risk_update
            if decision.exit is not None and symbol in positions:
                pending_exits[symbol] = decision.exit.reason
            if (
                decision.entry is not None
                and symbol not in positions
                and symbol not in pending_entries
            ):
                observed_bars = (
                    location + 1
                    if entry_history_start_date is None
                    else int(
                        (
                            (frame.index >= entry_history_start_date)
                            & (frame.index <= date)
                        ).sum()
                    )
                )
                if observed_bars < simulation_config.min_entry_history_bars:
                    diagnostics["insufficient_entry_history_count"] += 1
                    continue
                pending_entries[symbol] = _PendingEntry(
                    symbol=symbol,
                    created_date=date,
                    intent=decision.entry,
                    risk_multiplier=(
                        float(
                            entry_risk_multiplier(
                                symbol,
                                date,
                                frame.loc[date],
                                decision.entry,
                            )
                        )
                        if entry_risk_multiplier is not None
                        else 1.0
                    ),
                )
                signals.append(
                    {
                        "date": _date_str(date),
                        "symbol": symbol,
                        "strength": round(float(decision.entry.strength), 8),
                        "order_type": decision.entry.order_type,
                        "trigger_price": decision.entry.trigger_price,
                    }
                )

        equity = _portfolio_value(date, cash, positions, frames)
        peak_equity = max(peak_equity, equity)
        gross = _gross_exposure(date, positions, frames)
        equity_curve.append(
            {
                "date": _date_str(date),
                "equity": round(equity, 6),
                "cash": round(cash, 6),
                "gross_exposure": round(gross / equity, 6) if equity else 0.0,
                "drawdown_pct": round((equity / peak_equity - 1) * 100, 6)
                if peak_equity
                else 0.0,
            }
        )

    diagnostics["strategy_states"] = strategy_states
    diagnostics["signal_count"] = len(signals)
    return SimulationResult(
        summary=_summary(equity_curve, trades, simulation_config.initial_cash),
        equity_curve=equity_curve,
        positions=_position_rows(calendar[-1], positions, frames),
        trades=trades,
        signal_events=signals,
        symbol_contributions=_contribution_rows(
            contributions,
            positions,
            calendar[-1],
            frames,
        ),
        diagnostics=diagnostics,
    )


def _normalized_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    return frame.sort_index()


def _entry_price(intent: EntryIntent, row: pd.Series) -> float | None:
    if intent.order_type == "next_open":
        return float(row["Open"])
    if intent.trigger_price is None or float(row["High"]) < intent.trigger_price:
        return None
    return max(float(row["Open"]), intent.trigger_price)


def _risk_for_fill(intent: EntryIntent, fill_price: float) -> RiskIntent | None:
    if intent.risk is not None:
        return intent.risk
    stop_pct = intent.metadata.get("stop_loss_pct")
    target_pct = intent.metadata.get("take_profit_pct")
    if stop_pct is None and target_pct is None:
        return None
    return RiskIntent(
        stop_price=fill_price * (1 - float(stop_pct) / 100)
        if stop_pct is not None
        else None,
        target_price=fill_price * (1 + float(target_pct) / 100)
        if target_pct is not None
        else None,
    )


def _exit_for_bar(
    row: pd.Series,
    position: _Position,
    pending_reason: str | None,
) -> tuple[str | None, float | None]:
    if position.holding_bars >= 1 and position.risk is not None:
        stop = position.risk.stop_price
        target = position.risk.target_price
        if stop is not None and float(row["Low"]) <= stop:
            return position.risk.stop_reason, min(float(row["Open"]), stop)
        if target is not None and float(row["High"]) >= target:
            return "take_profit", target
    if pending_reason is not None:
        return pending_reason, float(row["Open"])
    return None, None


def _previous_close(frame: pd.DataFrame, date: pd.Timestamp) -> float:
    previous = frame[frame.index < date]
    return float(previous.iloc[-1]["Close"]) if not previous.empty else float(frame.loc[date, "Open"])


def _row_at(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    if date in frame.index:
        return frame.loc[date]
    previous = frame[frame.index <= date]
    return None if previous.empty else previous.iloc[-1]


def _gross_exposure(
    date: pd.Timestamp,
    positions: dict[str, _Position],
    frames: dict[str, pd.DataFrame],
) -> float:
    total = 0.0
    for symbol, position in positions.items():
        row = _row_at(frames[symbol], date)
        if row is not None:
            total += position.shares * float(row["Close"])
    return total


def _gross_exposure_at_execution(
    date: pd.Timestamp,
    positions: dict[str, _Position],
    frames: dict[str, pd.DataFrame],
) -> float:
    total = 0.0
    for symbol, position in positions.items():
        frame = frames[symbol]
        if date in frame.index:
            price = float(frame.loc[date, "Open"])
        else:
            row = _row_at(frame, date)
            if row is None:
                continue
            price = float(row["Close"])
        total += position.shares * price
    return total


def _portfolio_value(
    date: pd.Timestamp,
    cash: float,
    positions: dict[str, _Position],
    frames: dict[str, pd.DataFrame],
) -> float:
    return cash + _gross_exposure(date, positions, frames)


def _portfolio_value_at_execution(
    date: pd.Timestamp,
    cash: float,
    positions: dict[str, _Position],
    frames: dict[str, pd.DataFrame],
) -> float:
    return cash + _gross_exposure_at_execution(date, positions, frames)


def _trade(date, symbol, side, shares, price, amount, cost, reason, pnl):
    return {
        "date": _date_str(date),
        "symbol": symbol,
        "side": side,
        "shares": int(shares),
        "price": round(float(price), 6),
        "amount": round(float(amount), 6),
        "cost": round(float(cost), 6),
        "reason": reason,
        "pnl": None if pnl is None else round(float(pnl), 6),
    }


def _summary(curve, trades, initial_cash):
    final = float(curve[-1]["equity"]) if curve else float(initial_cash)
    total_return = (final / initial_cash - 1) * 100 if initial_cash else 0.0
    years = max(
        (pd.Timestamp(curve[-1]["date"]) - pd.Timestamp(curve[0]["date"])).days
        / 365.25,
        1 / 365.25,
    ) if curve else 1
    annual = ((final / initial_cash) ** (1 / years) - 1) * 100 if initial_cash and final > 0 else -100.0
    returns = pd.Series([point["equity"] for point in curve], dtype="float64").pct_change().dropna()
    sharpe = (
        float(returns.mean() / returns.std() * math.sqrt(252))
        if len(returns) > 1 and float(returns.std()) > 0
        else 0.0
    )
    sells = [trade for trade in trades if trade["side"] == "sell"]
    wins = [trade for trade in sells if float(trade.get("pnl") or 0) > 0]
    exposed_bars = sum(
        1 for point in curve if float(point.get("gross_exposure", 0)) > 0
    )
    return {
        "final_equity": round(final, 6),
        "total_return_pct": round(total_return, 6),
        "annual_return_pct": round(annual, 6),
        "max_drawdown_pct": round(
            abs(min((point["drawdown_pct"] for point in curve), default=0)),
            6,
        ),
        "sharpe": round(sharpe, 6),
        "trades": len(sells),
        "orders": len(trades),
        "win_rate_pct": round(len(wins) / len(sells) * 100, 6) if sells else 0.0,
        "turnover": round(sum(trade["amount"] for trade in trades) / initial_cash, 6)
        if initial_cash
        else 0.0,
        "exposure_time_pct": round(exposed_bars / len(curve) * 100, 6)
        if curve
        else 0.0,
        "final_gross_exposure": curve[-1]["gross_exposure"] if curve else 0.0,
    }


def _position_rows(date, positions, frames):
    rows = []
    for symbol, position in positions.items():
        row = _row_at(frames[symbol], date)
        if row is None:
            continue
        price = float(row["Close"])
        rows.append(
            {
                "symbol": symbol,
                "shares": position.shares,
                "entry_date": position.entry_date,
                "entry_price": position.entry_price,
                "price": round(price, 6),
                "market_value": round(position.shares * price, 6),
                "unrealized_pnl": round(
                    position.shares * (price - position.entry_price) - position.entry_cost,
                    6,
                ),
                "holding_bars": position.holding_bars,
            }
        )
    return rows


def _contribution_rows(contributions, positions, date, frames):
    rows = []
    for symbol in sorted(contributions):
        bucket = contributions[symbol]
        unrealized = 0.0
        if symbol in positions:
            position = positions[symbol]
            row = _row_at(frames[symbol], date)
            if row is not None:
                unrealized = (
                    position.shares * (float(row["Close"]) - position.entry_price)
                    - position.entry_cost
                )
        realized = float(bucket["realized_pnl"])
        rows.append(
            {
                "symbol": symbol,
                "round_trips": int(bucket["round_trips"]),
                "realized_pnl": round(realized, 6),
                "unrealized_pnl": round(unrealized, 6),
                "total_pnl": round(realized + unrealized, 6),
            }
        )
    return sorted(rows, key=lambda row: row["total_pnl"], reverse=True)


def _emit(callback, **event):
    if callback is not None:
        callback(event)


def _date_str(date) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")
