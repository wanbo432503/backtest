from __future__ import annotations

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict

from optimization_models import AShareTradingConfig
from strategy_engine import (
    EntryIntent,
    ExitIntent,
    RiskIntent,
    StrategyDecision,
    StrategyDefinition,
)
from strategy_simulator import SimulationConfig, run_strategy_simulation


class FakeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _data(
    opens=(10.0, 10.0, 10.0, 10.0),
    highs=None,
    lows=None,
    closes=None,
) -> pd.DataFrame:
    count = len(opens)
    highs = highs or tuple(value + 1 for value in opens)
    lows = lows or tuple(value - 1 for value in opens)
    closes = closes or opens
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1_000_000] * count,
        },
        index=pd.date_range("2026-01-01", periods=count, freq="D"),
    )


def _trading(**overrides) -> AShareTradingConfig:
    values = {
        "t_plus_one": True,
        "lot_size": 1,
        "limit_up_down_filter": False,
        "volume_filter": False,
        "slippage_pct": 0,
        "buy_commission_pct": 0,
        "sell_commission_pct": 0,
        "stamp_tax_pct": 0,
        "min_commission": 0,
    }
    return AShareTradingConfig(**{**values, **overrides})


def _simulation(**overrides) -> SimulationConfig:
    values = {
        "initial_cash": 10_000,
        "max_positions": 10,
        "max_position_pct": 1.0,
        "target_gross_exposure": 1.0,
        "max_drawdown_stop_pct": None,
        "trading": _trading(),
    }
    return SimulationConfig(**{**values, **overrides})


def _definition(evaluator) -> StrategyDefinition:
    return StrategyDefinition(
        strategy_id="fake",
        display_name="Fake",
        description="Fake strategy",
        config_model=FakeConfig,
        parameters=(),
        prepare_frame=lambda data, config: data.copy(),
        evaluate=evaluator,
        min_history_bars=lambda config: 1,
    )


def test_close_signal_fills_at_next_open_without_lookahead():
    definition = _definition(
        lambda context: StrategyDecision(
            entry=EntryIntent("next_open", suggested_position_pct=0.5)
        )
        if context.bar_index == 0 and context.position is None
        else StrategyDecision()
    )

    result = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": _data(opens=(10, 11, 12))},
        _simulation(),
    )

    buy = result.trades[0]
    assert buy["side"] == "buy"
    assert buy["date"] == "2026-01-02"
    assert buy["price"] == 11


def test_stop_next_bar_only_fills_when_next_high_reaches_trigger():
    definition = _definition(
        lambda context: StrategyDecision(
            entry=EntryIntent(
                "stop_next_bar",
                trigger_price=11,
                expires_after_bars=1,
            )
        )
        if context.bar_index == 0
        else StrategyDecision()
    )

    missed = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": _data(opens=(10, 10), highs=(10.5, 10.9))},
        _simulation(),
    )
    filled = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": _data(opens=(10, 10), highs=(10.5, 11.2))},
        _simulation(),
    )

    assert missed.trades == []
    assert filled.trades[0]["price"] == 11


def test_pending_order_expires_after_declared_bars():
    definition = _definition(
        lambda context: StrategyDecision(
            entry=EntryIntent("stop_next_bar", trigger_price=11, expires_after_bars=1)
        )
        if context.bar_index == 0
        else StrategyDecision()
    )
    data = _data(opens=(10, 10, 10), highs=(10.5, 10.9, 11.5))

    result = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": data},
        _simulation(),
    )

    assert result.trades == []
    assert result.diagnostics["expired_entry_count"] == 1


def test_t1_blocks_exit_until_trading_day_after_fill():
    def evaluator(context):
        if context.position is None and context.bar_index == 0:
            return StrategyDecision(entry=EntryIntent("next_open"))
        if context.position is not None:
            return StrategyDecision(exit=ExitIntent("signal_exit"))
        return StrategyDecision()

    result = run_strategy_simulation(
        _definition(evaluator),
        FakeConfig(),
        {"SH603019": _data(opens=(10, 10, 10, 10))},
        _simulation(),
    )

    sell = [trade for trade in result.trades if trade["side"] == "sell"][0]
    assert sell["date"] == "2026-01-03"


def test_shared_cash_allocates_stronger_signal_first():
    def evaluator(context):
        if context.bar_index != 0:
            return StrategyDecision()
        strength = 2 if context.symbol == "SH603019" else 1
        return StrategyDecision(entry=EntryIntent("next_open", strength=strength))

    result = run_strategy_simulation(
        _definition(evaluator),
        FakeConfig(),
        {
            "SH603019": _data(opens=(10, 10)),
            "SZ002241": _data(opens=(10, 10)),
        },
        _simulation(initial_cash=1_000, max_positions=2),
    )

    buys = [trade for trade in result.trades if trade["side"] == "buy"]
    assert [trade["symbol"] for trade in buys] == ["SH603019"]


def test_max_positions_is_a_hard_cap():
    definition = _definition(
        lambda context: StrategyDecision(entry=EntryIntent("next_open"))
        if context.bar_index == 0
        else StrategyDecision()
    )
    result = run_strategy_simulation(
        definition,
        FakeConfig(),
        {
            "SH603019": _data(opens=(10, 10)),
            "SZ002241": _data(opens=(10, 10)),
        },
        _simulation(max_positions=1),
    )

    assert len([trade for trade in result.trades if trade["side"] == "buy"]) == 1


def test_strategy_suggested_size_is_capped_by_portfolio_limit():
    definition = _definition(
        lambda context: StrategyDecision(
            entry=EntryIntent("next_open", suggested_position_pct=0.8)
        )
        if context.bar_index == 0
        else StrategyDecision()
    )
    result = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": _data(opens=(10, 10))},
        _simulation(initial_cash=1_000, max_position_pct=0.25),
    )

    buy = result.trades[0]
    assert buy["amount"] <= 250


def test_lot_rounding_and_trade_costs_are_applied():
    definition = _definition(
        lambda context: StrategyDecision(
            entry=EntryIntent("next_open", suggested_position_pct=0.5)
        )
        if context.bar_index == 0
        else StrategyDecision()
    )
    result = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": _data(opens=(10, 10))},
        _simulation(
            initial_cash=2_000,
            trading=_trading(lot_size=100, min_commission=5),
        ),
    )

    buy = result.trades[0]
    assert buy["shares"] == 100
    assert buy["cost"] == 5


def test_risk_budget_caps_position_size():
    definition = _definition(
        lambda context: StrategyDecision(
            entry=EntryIntent(
                "next_open",
                risk=RiskIntent(risk_per_share=2, risk_budget_pct=0.01),
            )
        )
        if context.bar_index == 0
        else StrategyDecision()
    )
    result = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": _data(opens=(10, 10))},
        _simulation(initial_cash=10_000),
    )

    assert result.trades[0]["shares"] == 50


def test_entry_overlay_is_frozen_on_signal_date():
    definition = _definition(
        lambda context: StrategyDecision(entry=EntryIntent("next_open"))
        if context.bar_index == 0
        else StrategyDecision()
    )

    result = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": _data(opens=(10, 10))},
        _simulation(initial_cash=10_000),
        entry_risk_multiplier=lambda symbol, date, row, intent: (
            0.5 if date == pd.Timestamp("2026-01-01") else 1.0
        ),
    )

    assert result.trades[0]["amount"] == 5_000


def test_drawdown_gate_is_evaluated_before_pending_entries():
    def evaluator(context):
        if context.symbol == "SH603019" and context.bar_index == 0:
            return StrategyDecision(
                entry=EntryIntent("next_open", suggested_position_pct=0.5)
            )
        if context.symbol == "SZ002241" and context.bar_index == 1:
            return StrategyDecision(entry=EntryIntent("next_open"))
        return StrategyDecision()

    result = run_strategy_simulation(
        _definition(evaluator),
        FakeConfig(),
        {
            "SH603019": _data(
                opens=(100, 100, 50, 50),
                closes=(100, 100, 50, 50),
            ),
            "SZ002241": _data(opens=(10, 10, 10, 10)),
        },
        _simulation(initial_cash=10_000, max_drawdown_stop_pct=20),
    )

    assert not any(
        trade["symbol"] == "SZ002241" and trade["side"] == "buy"
        for trade in result.trades
    )
    assert result.diagnostics["drawdown_entry_stop"] is True


def test_same_bar_stop_and_target_uses_conservative_stop_first():
    definition = _definition(
        lambda context: StrategyDecision(
            entry=EntryIntent(
                "next_open",
                risk=RiskIntent(stop_price=95, target_price=105),
            )
        )
        if context.bar_index == 0
        else StrategyDecision()
    )
    data = _data(
        opens=(100, 100, 100),
        highs=(101, 101, 106),
        lows=(99, 99, 94),
    )

    result = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": data},
        _simulation(),
    )

    sell = [trade for trade in result.trades if trade["side"] == "sell"][0]
    assert sell["reason"] == "stop_loss"
    assert sell["price"] == 95


def test_strategy_state_is_replaced_without_mutating_prior_mapping():
    seen_states = []

    def evaluator(context):
        seen_states.append(context.state)
        next_state = {"count": int(context.state.get("count", 0)) + 1}
        return StrategyDecision(next_state=next_state)

    result = run_strategy_simulation(
        _definition(evaluator),
        FakeConfig(),
        {"SH603019": _data(opens=(10, 10, 10))},
        _simulation(),
    )

    assert [dict(state) for state in seen_states] == [{}, {"count": 1}, {"count": 2}]
    assert len({id(state) for state in seen_states}) == 3
    assert result.diagnostics["strategy_states"]["SH603019"] == {"count": 3}


def test_context_reports_completed_exit_cooldown_for_reentry_rules():
    def evaluator(context):
        if context.position is not None:
            return StrategyDecision(exit=ExitIntent("rotate"))
        if context.bars_since_exit is None or context.bars_since_exit > 2:
            return StrategyDecision(entry=EntryIntent("next_open"))
        return StrategyDecision()

    result = run_strategy_simulation(
        _definition(evaluator),
        FakeConfig(),
        {"SH603019": _data(opens=(10, 10, 10, 10, 10, 10, 10))},
        _simulation(),
    )

    buys = [trade for trade in result.trades if trade["side"] == "buy"]
    assert [trade["date"] for trade in buys] == ["2026-01-02", "2026-01-07"]


def test_future_price_changes_do_not_change_earlier_fills():
    definition = _definition(
        lambda context: StrategyDecision(
            entry=EntryIntent("next_open", suggested_position_pct=0.5)
        )
        if context.bar_index == 0
        else StrategyDecision()
    )
    original = _data(opens=(10, 11, 12, 13))
    changed = original.copy()
    changed.iloc[2:, changed.columns.get_loc("Open")] = [1_000, 2_000]
    changed.iloc[2:, changed.columns.get_loc("Close")] = [1_000, 2_000]

    first = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": original},
        _simulation(),
    )
    second = run_strategy_simulation(
        definition,
        FakeConfig(),
        {"SH603019": changed},
        _simulation(),
    )

    assert first.trades[0] == second.trades[0]
    assert first.equity_curve[:2] == second.equity_curve[:2]


def test_fill_day_close_does_not_change_later_entry_size():
    def evaluator(context):
        if context.symbol == "SH603019" and context.bar_index == 0:
            return StrategyDecision(
                entry=EntryIntent("next_open", suggested_position_pct=0.5)
            )
        if context.symbol == "SZ002241" and context.bar_index == 1:
            return StrategyDecision(
                entry=EntryIntent("next_open", suggested_position_pct=0.25)
            )
        return StrategyDecision()

    first_symbol = _data(opens=(100, 100, 100), closes=(100, 100, 100))
    changed_close = first_symbol.copy()
    changed_close.loc[pd.Timestamp("2026-01-03"), "Close"] = 1_000
    second_symbol = _data(opens=(10, 10, 10), closes=(10, 10, 10))

    original = run_strategy_simulation(
        _definition(evaluator),
        FakeConfig(),
        {"SH603019": first_symbol, "SZ002241": second_symbol},
        _simulation(initial_cash=10_000, max_positions=2),
    )
    changed = run_strategy_simulation(
        _definition(evaluator),
        FakeConfig(),
        {"SH603019": changed_close, "SZ002241": second_symbol},
        _simulation(initial_cash=10_000, max_positions=2),
    )

    original_buy = [trade for trade in original.trades if trade["symbol"] == "SZ002241"]
    changed_buy = [trade for trade in changed.trades if trade["symbol"] == "SZ002241"]
    assert original_buy == changed_buy


def test_strategy_preparation_error_includes_strategy_and_symbol_context():
    definition = _definition(lambda context: StrategyDecision())
    definition = StrategyDefinition(
        **{
            **definition.__dict__,
            "prepare_frame": lambda data, config: (_ for _ in ()).throw(
                RuntimeError("indicator failed")
            ),
        }
    )

    with pytest.raises(
        RuntimeError,
        match="strategy 'fake'.*SH603019.*preparation",
    ):
        run_strategy_simulation(
            definition,
            FakeConfig(),
            {"SH603019": _data()},
            _simulation(),
        )


def test_strategy_evaluation_error_includes_strategy_symbol_and_date_context():
    definition = _definition(
        lambda context: (_ for _ in ()).throw(RuntimeError("decision failed"))
    )

    with pytest.raises(
        RuntimeError,
        match="strategy 'fake'.*SH603019.*2026-01-01.*evaluation",
    ):
        run_strategy_simulation(
            definition,
            FakeConfig(),
            {"SH603019": _data()},
            _simulation(),
        )
