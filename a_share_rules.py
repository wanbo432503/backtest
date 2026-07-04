from typing import Any

from optimization_models import AShareTradingConfig


def is_limit_up(current_close: float, previous_close: float, threshold: float = 0.1) -> bool:
    if previous_close <= 0:
        return False
    return current_close / previous_close - 1 >= threshold - 1e-9


def is_limit_down(current_close: float, previous_close: float, threshold: float = 0.1) -> bool:
    if previous_close <= 0:
        return False
    return current_close / previous_close - 1 <= -threshold + 1e-9


def round_lot_shares(raw_shares: float, lot_size: int = 100) -> int:
    if raw_shares <= 0:
        return 0
    return int(raw_shares // lot_size * lot_size)


def can_buy(
    row: dict[str, Any],
    previous_close: float,
    config: AShareTradingConfig,
) -> tuple[bool, str | None]:
    close = float(row.get("Close", row.get("close", 0)))
    volume = float(row.get("Volume", row.get("volume", 0)))
    if config.limit_up_down_filter and is_limit_up(close, previous_close):
        return False, "limit_up"
    if config.volume_filter and volume < config.min_volume:
        return False, "low_volume"
    return True, None


def can_sell(
    row: dict[str, Any],
    previous_close: float,
    holding_bars: int,
    config: AShareTradingConfig,
) -> tuple[bool, str | None]:
    close = float(row.get("Close", row.get("close", 0)))
    if config.t_plus_one and holding_bars < 1:
        return False, "t_plus_one"
    if config.limit_up_down_filter and is_limit_down(close, previous_close):
        return False, "limit_down"
    return True, None


def apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    rate = slippage_pct / 100
    if side == "buy":
        return round(price * (1 + rate), 6)
    if side == "sell":
        return round(price * (1 - rate), 6)
    raise ValueError("side must be 'buy' or 'sell'")


def calculate_trade_cost(amount: float, side: str, config: AShareTradingConfig) -> float:
    if side == "buy":
        commission = max(amount * config.buy_commission_pct / 100, config.min_commission)
        return round(commission, 6)
    if side == "sell":
        commission = max(amount * config.sell_commission_pct / 100, config.min_commission)
        stamp_tax = amount * config.stamp_tax_pct / 100
        return round(commission + stamp_tax, 6)
    raise ValueError("side must be 'buy' or 'sell'")
