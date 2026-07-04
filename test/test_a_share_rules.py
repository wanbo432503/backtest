from optimization_models import AShareTradingConfig
from a_share_rules import (
    apply_slippage,
    calculate_trade_cost,
    can_buy,
    can_sell,
    is_limit_down,
    is_limit_up,
    round_lot_shares,
)


def test_detects_limit_up_and_limit_down():
    assert is_limit_up(current_close=11, previous_close=10)
    assert is_limit_down(current_close=9, previous_close=10)


def test_rounds_shares_down_to_a_share_lot_size():
    assert round_lot_shares(356, lot_size=100) == 300
    assert round_lot_shares(99, lot_size=100) == 0


def test_buy_slippage_increases_execution_price():
    assert apply_slippage(10, side="buy", slippage_pct=0.05) == 10.005


def test_sell_slippage_decreases_execution_price():
    assert apply_slippage(10, side="sell", slippage_pct=0.05) == 9.995


def test_buy_trade_cost_uses_commission_and_minimum_fee():
    config = AShareTradingConfig(buy_commission_pct=0.03, min_commission=5)

    assert calculate_trade_cost(10000, side="buy", config=config) == 5
    assert calculate_trade_cost(100000, side="buy", config=config) == 30


def test_sell_trade_cost_includes_commission_and_stamp_tax():
    config = AShareTradingConfig(
        sell_commission_pct=0.03,
        stamp_tax_pct=0.05,
        min_commission=5,
    )

    assert calculate_trade_cost(100000, side="sell", config=config) == 80


def test_can_buy_blocks_limit_up_and_low_volume():
    config = AShareTradingConfig(limit_up_down_filter=True, volume_filter=True, min_volume=1000)

    assert can_buy({"Close": 11, "Volume": 2000}, previous_close=10, config=config) == (
        False,
        "limit_up",
    )
    assert can_buy({"Close": 10.2, "Volume": 500}, previous_close=10, config=config) == (
        False,
        "low_volume",
    )


def test_can_sell_blocks_t_plus_one_same_bar_and_limit_down():
    config = AShareTradingConfig(t_plus_one=True, limit_up_down_filter=True)

    assert can_sell({"Close": 10.1, "Volume": 2000}, previous_close=10, holding_bars=0, config=config) == (
        False,
        "t_plus_one",
    )
    assert can_sell({"Close": 9, "Volume": 2000}, previous_close=10, holding_bars=1, config=config) == (
        False,
        "limit_down",
    )
