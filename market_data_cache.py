from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any

import pandas as pd


CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / "data" / "market_cache" / "daily"
_LOCKS: dict[str, Lock] = {}
_LOCKS_GUARD = Lock()


@dataclass
class DailyCacheSnapshot:
    symbol: str
    provider: str
    data: pd.DataFrame
    covered_ranges: list[tuple[date, date]]
    warnings: list[str]

    def covers(self, start_date: str, end_date: str) -> bool:
        return not uncovered_date_ranges(self.covered_ranges, start_date, end_date)


def daily_cache_enabled() -> bool:
    value = os.environ.get("MARKET_DATA_CACHE_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def cache_root() -> Path:
    configured = os.environ.get("MARKET_DATA_CACHE_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_CACHE_ROOT


def load_daily_cache(symbol: str, provider: str) -> DailyCacheSnapshot | None:
    data_path, metadata_path = _cache_paths(symbol, provider)
    if not data_path.exists() or not metadata_path.exists():
        return None
    lock = _lock_for(data_path)
    with lock:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
                return None
            frame = pd.read_csv(data_path, index_col="Date", parse_dates=["Date"])
            frame.index = pd.to_datetime(frame.index).normalize()
            ranges = [
                (date.fromisoformat(item["start"]), date.fromisoformat(item["end"]))
                for item in metadata.get("covered_ranges", [])
            ]
            return DailyCacheSnapshot(
                symbol=str(metadata.get("symbol") or symbol),
                provider=str(metadata.get("provider") or provider),
                data=frame.sort_index(),
                covered_ranges=_merge_ranges(ranges),
                warnings=[str(value) for value in metadata.get("warnings", [])],
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None


def save_daily_cache(
    symbol: str,
    provider: str,
    data: pd.DataFrame,
    covered_ranges: list[tuple[date, date]],
    warnings: list[str],
) -> DailyCacheSnapshot:
    data_path, metadata_path = _cache_paths(symbol, provider)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_for(data_path)
    with lock:
        existing = _load_without_lock(data_path, metadata_path, symbol, provider)
        frames = [frame for frame in [existing.data if existing else None, data] if frame is not None]
        merged = pd.concat(frames).sort_index() if frames else pd.DataFrame()
        if not merged.empty:
            merged = merged[~merged.index.duplicated(keep="last")]
            if isinstance(merged.index, pd.DatetimeIndex) and merged.index.tz is not None:
                merged.index = merged.index.tz_localize(None)
            merged.index = pd.to_datetime(merged.index).normalize()
            merged.index.name = "Date"
        ranges = _merge_ranges([
            *(existing.covered_ranges if existing else []),
            *covered_ranges,
        ])
        combined_warnings = list(dict.fromkeys([
            *(existing.warnings if existing else []),
            *[str(value) for value in warnings],
        ]))
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "symbol": symbol,
            "provider": provider,
            "covered_ranges": [
                {"start": start.isoformat(), "end": end.isoformat()}
                for start, end in ranges
            ],
            "warnings": combined_warnings,
        }
        _atomic_write_csv(data_path, merged)
        _atomic_write_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2))
        return DailyCacheSnapshot(symbol, provider, merged, ranges, combined_warnings)


def uncovered_date_ranges(
    covered_ranges: list[tuple[date, date]],
    start_date: str,
    end_date: str,
) -> list[tuple[date, date]]:
    requested_start = date.fromisoformat(start_date)
    requested_end = date.fromisoformat(end_date)
    if requested_start > requested_end:
        raise ValueError("cache start_date must not be after end_date")
    gaps: list[tuple[date, date]] = []
    cursor = requested_start
    for covered_start, covered_end in _merge_ranges(covered_ranges):
        if covered_end < cursor or covered_start > requested_end:
            continue
        if covered_start > cursor:
            gaps.append((cursor, min(covered_start - timedelta(days=1), requested_end)))
        cursor = max(cursor, covered_end + timedelta(days=1))
        if cursor > requested_end:
            break
    if cursor <= requested_end:
        gaps.append((cursor, requested_end))
    return gaps


def slice_daily_cache(snapshot: DailyCacheSnapshot, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return snapshot.data[(snapshot.data.index >= start) & (snapshot.data.index < end)].copy()


def _cache_paths(symbol: str, provider: str) -> tuple[Path, Path]:
    safe_symbol = "".join(character for character in symbol.lower() if character.isalnum() or character in "-_")
    safe_provider = "".join(character for character in provider.lower() if character.isalnum() or character in "-_")
    base = cache_root() / safe_provider / safe_symbol
    return base.with_suffix(".csv"), base.with_suffix(".json")


def _load_without_lock(
    data_path: Path,
    metadata_path: Path,
    symbol: str,
    provider: str,
) -> DailyCacheSnapshot | None:
    if not data_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        frame = pd.read_csv(data_path, index_col="Date", parse_dates=["Date"])
        frame.index = pd.to_datetime(frame.index).normalize()
        ranges = [
            (date.fromisoformat(item["start"]), date.fromisoformat(item["end"]))
            for item in metadata.get("covered_ranges", [])
        ]
        return DailyCacheSnapshot(
            symbol,
            provider,
            frame,
            _merge_ranges(ranges),
            [str(value) for value in metadata.get("warnings", [])],
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _merge_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    merged: list[tuple[date, date]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + timedelta(days=1):
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _lock_for(path: Path) -> Lock:
    key = str(path)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, Lock())


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle)
    temporary.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(path)
