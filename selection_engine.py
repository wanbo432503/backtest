from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from portfolio_models import RebalanceConfig, SelectionConfig


@dataclass
class SelectionResult:
    selected: list[dict[str, Any]] = field(default_factory=list)
    ranking: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_trading_calendar(data_by_symbol: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for data in data_by_symbol.values():
        dates.update(pd.Timestamp(index_value).normalize() for index_value in data.index)
    return sorted(dates)


def build_rebalance_dates(
    calendar: list[pd.Timestamp],
    start_date: str,
    end_date: str,
    config: RebalanceConfig,
) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    available_dates = [pd.Timestamp(date).normalize() for date in calendar if start <= date <= end]

    if config.frequency == "weekly":
        return _weekly_dates(available_dates, config.weekday, start)
    if config.frequency == "biweekly":
        return _weekly_dates(available_dates, config.weekday, start)[::2]
    if config.frequency == "monthly":
        return _monthly_dates(available_dates, config.monthday)
    return []


def select_top_candidates(
    candidate_rows: list[dict[str, Any]],
    selection_config: SelectionConfig,
) -> SelectionResult:
    ranking = sorted(
        candidate_rows,
        key=lambda row: (
            row.get("skip_reason") is not None,
            -(row.get("score") if row.get("score") is not None else float("-inf")),
            row.get("symbol", ""),
        ),
    )
    selectable = [
        row
        for row in ranking
        if row.get("skip_reason") is None
        and row.get("score") is not None
        and (
            selection_config.score_threshold is None
            or float(row.get("score", 0)) >= selection_config.score_threshold
        )
    ]
    selected = selectable[: selection_config.top_n]
    warnings = []
    if len(selected) < selection_config.top_n:
        warnings.append(f"selected {len(selected)} candidates, below top_n {selection_config.top_n}")
    return SelectionResult(selected=selected, ranking=ranking, warnings=warnings)


def _weekly_dates(
    calendar: list[pd.Timestamp],
    weekday: int,
    start: pd.Timestamp,
) -> list[pd.Timestamp]:
    by_week: dict[tuple[int, int], pd.Timestamp] = {}
    for date in calendar:
        if date.weekday() < weekday:
            continue
        target_date = date - pd.Timedelta(days=date.weekday() - weekday)
        if target_date < start.normalize():
            continue
        year_week = date.isocalendar()[:2]
        by_week.setdefault(year_week, date)
    return sorted(by_week.values())


def _monthly_dates(calendar: list[pd.Timestamp], monthday: int) -> list[pd.Timestamp]:
    by_month: dict[tuple[int, int], pd.Timestamp] = {}
    for date in calendar:
        if date.day < monthday:
            continue
        key = (date.year, date.month)
        by_month.setdefault(key, date)
    return sorted(by_month.values())
