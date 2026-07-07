from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TradableUniversePolicy:
    max_symbols: int = 4
    allowed_code_prefixes: tuple[str, ...] = ("60", "00")
    exclude_funds: bool = True


@dataclass(frozen=True)
class TradableSymbolResult:
    raw: str
    symbol: str | None
    normalized_symbol: str | None
    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class UniverseValidationResult:
    accepted_symbols: list[str] = field(default_factory=list)
    rejected: list[TradableSymbolResult] = field(default_factory=list)
    ok: bool = True


def normalize_tradable_symbol(value: str) -> str | None:
    parsed = _parse_symbol(value)
    if parsed is None:
        return None
    exchange, code = parsed
    return f"{exchange}{code}"


def validate_tradable_symbol(
    value: str,
    policy: TradableUniversePolicy | None = None,
) -> TradableSymbolResult:
    active_policy = policy or TradableUniversePolicy()
    raw = str(value or "").strip()
    parsed = _parse_symbol(raw)
    if parsed is None:
        return TradableSymbolResult(raw=raw, symbol=None, normalized_symbol=None, ok=False, reason="not_a_share")

    exchange, code = parsed
    normalized = f"{exchange}{code}"
    reason = _blocked_reason(exchange, code, active_policy)
    if reason:
        return TradableSymbolResult(raw=raw, symbol=normalized, normalized_symbol=normalized, ok=False, reason=reason)

    return TradableSymbolResult(raw=raw, symbol=normalized, normalized_symbol=normalized, ok=True)


def validate_universe(
    symbols: list[str],
    policy: TradableUniversePolicy | None = None,
) -> UniverseValidationResult:
    active_policy = policy or TradableUniversePolicy()
    accepted: list[str] = []
    rejected: list[TradableSymbolResult] = []
    seen: set[str] = set()
    has_blocking_error = False

    for symbol in symbols:
        result = validate_tradable_symbol(symbol, active_policy)
        if not result.ok:
            rejected.append(result)
            has_blocking_error = True
            continue

        normalized = result.normalized_symbol or ""
        if normalized in seen:
            rejected.append(
                TradableSymbolResult(
                    raw=result.raw,
                    symbol=normalized,
                    normalized_symbol=normalized,
                    ok=False,
                    reason="duplicate_symbol",
                )
            )
            continue

        seen.add(normalized)
        accepted.append(normalized)

    if len(accepted) > active_policy.max_symbols:
        rejected.append(
            TradableSymbolResult(
                raw=",".join(accepted),
                symbol=None,
                normalized_symbol=None,
                ok=False,
                reason="too_many_symbols",
            )
        )
        has_blocking_error = True

    if not accepted:
        has_blocking_error = True

    return UniverseValidationResult(
        accepted_symbols=accepted,
        rejected=rejected,
        ok=not has_blocking_error,
    )


def _parse_symbol(value: str) -> tuple[str, str] | None:
    normalized = str(value or "").strip().upper()
    if not normalized:
        return None

    if len(normalized) == 8 and normalized[:2] in {"SH", "SZ", "BJ"} and normalized[2:].isdigit():
        return normalized[:2], normalized[2:]

    if "." in normalized:
        code, suffix = normalized.split(".", 1)
        if suffix in {"SH", "SZ", "BJ"} and code.isdigit() and len(code) == 6:
            return suffix, code
        return None

    if normalized.isdigit() and len(normalized) == 6:
        if normalized.startswith(("6", "5")):
            return "SH", normalized
        if normalized.startswith(("0", "1", "2", "3")):
            return "SZ", normalized
        if normalized.startswith(("4", "8", "9")):
            return "BJ", normalized

    return None


def _blocked_reason(exchange: str, code: str, policy: TradableUniversePolicy) -> str | None:
    if exchange == "BJ" or code.startswith(("4", "8", "9")):
        return "unsupported_board"

    if exchange == "SH" and code.startswith(("688", "689")):
        return "unsupported_board"

    if exchange == "SZ" and code.startswith(("300", "301")):
        return "unsupported_board"

    if policy.exclude_funds and _looks_like_fund_or_etf(exchange, code):
        return "fund_or_etf"

    if not code.startswith(policy.allowed_code_prefixes):
        return "only_60_00_prefix"

    return None


def _looks_like_fund_or_etf(exchange: str, code: str) -> bool:
    if exchange == "SH" and code.startswith(("510", "511", "512")):
        return True
    if exchange == "SZ" and code.startswith(("159", "160")):
        return True
    return False
