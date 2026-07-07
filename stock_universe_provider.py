from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from tradable_universe import validate_tradable_symbol


DEFAULT_UNIVERSE_CACHE_PATH = Path("data/stock_universe_60_00.json")


@dataclass(frozen=True)
class StockUniverseRecord:
    symbol: str
    name: str = ""
    exchange: str = ""
    code_prefix: str = ""
    status: str = "active"
    source: str = "unknown"
    refreshed_at: str = ""


@dataclass(frozen=True)
class StockUniverseResult:
    records: list[StockUniverseRecord] = field(default_factory=list)
    source: str = "unknown"
    warnings: list[str] = field(default_factory=list)


def get_default_stock_universe(
    *,
    cache_path: Path | str = DEFAULT_UNIVERSE_CACHE_PATH,
    refresh: bool = False,
    fetcher: Callable[[], Iterable[StockUniverseRecord]] | None = None,
    max_symbols: int | None = None,
) -> StockUniverseResult:
    path = Path(cache_path)
    warnings: list[str] = []

    if not refresh:
        cached = load_cached_universe(path)
        if cached:
            return StockUniverseResult(
                records=_limit_records(cached, max_symbols),
                source="cache",
                warnings=[],
            )

    active_fetcher = fetcher or fetch_mootdx_universe
    try:
        fetched = filter_tradable_universe_records(list(active_fetcher()))
        if fetched:
            save_cached_universe(fetched, path)
            return StockUniverseResult(
                records=_limit_records(fetched, max_symbols),
                source=fetched[0].source or "remote",
                warnings=[],
            )
    except Exception as exc:
        warnings.append(f"universe_fetch_failed: {exc}")

    fallback = filter_tradable_universe_records(_builtin_universe_records())
    if fallback:
        save_cached_universe(fallback, path)
    return StockUniverseResult(
        records=_limit_records(fallback, max_symbols),
        source="builtin",
        warnings=warnings,
    )


def filter_tradable_universe_records(
    records: Iterable[StockUniverseRecord],
) -> list[StockUniverseRecord]:
    filtered: list[StockUniverseRecord] = []
    seen: set[str] = set()
    for record in records:
        validation = validate_tradable_symbol(record.symbol)
        if not validation.ok or not validation.normalized_symbol:
            continue
        normalized = validation.normalized_symbol
        if normalized in seen:
            continue
        seen.add(normalized)
        filtered.append(_normalize_record(record, normalized))
    return filtered


def load_cached_universe(cache_path: Path | str) -> list[StockUniverseRecord]:
    path = Path(cache_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("records", payload if isinstance(payload, list) else [])
    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            records.append(StockUniverseRecord(**row))
        except TypeError:
            continue
    return filter_tradable_universe_records(records)


def save_cached_universe(records: Iterable[StockUniverseRecord], cache_path: Path | str) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "refreshed_at": _utc_now(),
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_mootdx_universe() -> list[StockUniverseRecord]:
    try:
        from mootdx.quotes import Quotes
    except ImportError as exc:
        raise RuntimeError("mootdx is not installed") from exc

    client = Quotes.factory(market="std")
    records: list[StockUniverseRecord] = []
    for market, exchange in ((1, "SH"), (0, "SZ")):
        frame = client.stocks(market=market)
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            code = str(row.get("code", "")).strip()
            if not code or not code.isdigit() or len(code) != 6:
                continue
            name = str(row.get("name", "") or "").strip()
            records.append(
                StockUniverseRecord(
                    symbol=f"{exchange}{code}",
                    name=name,
                    exchange=exchange,
                    code_prefix=code[:2],
                    source="mootdx",
                    refreshed_at=_utc_now(),
                )
            )
    return records


def _builtin_universe_records() -> list[StockUniverseRecord]:
    try:
        from stock_search import COMMON_CN_STOCKS
    except Exception:
        COMMON_CN_STOCKS = {}

    records = [
        _record_from_symbol(symbol, name=name, source="builtin")
        for name, symbol in COMMON_CN_STOCKS.items()
    ]
    if records:
        return records

    fallback_symbols = {
        "浦发银行": "SH600000",
        "中国平安": "SH601318",
        "中科曙光": "SH603019",
        "平安银行": "SZ000001",
        "万科A": "SZ000002",
        "歌尔股份": "SZ002241",
    }
    return [
        _record_from_symbol(symbol, name=name, source="builtin")
        for name, symbol in fallback_symbols.items()
    ]


def _record_from_symbol(symbol: str, *, name: str = "", source: str = "unknown") -> StockUniverseRecord:
    normalized = validate_tradable_symbol(symbol).normalized_symbol or str(symbol).strip().upper()
    exchange = normalized[:2] if len(normalized) >= 2 else ""
    code = normalized[2:] if len(normalized) >= 8 else ""
    return StockUniverseRecord(
        symbol=normalized,
        name=name,
        exchange=exchange,
        code_prefix=code[:2],
        source=source,
        refreshed_at=_utc_now(),
    )


def _normalize_record(record: StockUniverseRecord, normalized_symbol: str) -> StockUniverseRecord:
    exchange = normalized_symbol[:2]
    code = normalized_symbol[2:]
    return StockUniverseRecord(
        symbol=normalized_symbol,
        name=record.name,
        exchange=record.exchange or exchange,
        code_prefix=record.code_prefix or code[:2],
        status=record.status or "active",
        source=record.source or "unknown",
        refreshed_at=record.refreshed_at or _utc_now(),
    )


def _limit_records(
    records: list[StockUniverseRecord],
    max_symbols: int | None,
) -> list[StockUniverseRecord]:
    if max_symbols is None:
        return records
    return records[:max(max_symbols, 0)]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
