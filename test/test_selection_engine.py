import pandas as pd

from portfolio_models import RebalanceConfig, SelectionConfig
from selection_engine import build_rebalance_dates, build_trading_calendar, select_top_candidates
from test.fixtures.portfolio_ohlcv import build_ohlcv_frame


def test_build_trading_calendar_uses_sorted_union_of_dates():
    first = build_ohlcv_frame(start_date="2025-01-01", periods=5)
    second = build_ohlcv_frame(start_date="2025-01-03", periods=5)

    calendar = build_trading_calendar({"SH603019": first, "SZ002241": second})

    assert calendar == sorted(set(first.index).union(set(second.index)))


def test_build_rebalance_dates_weekly_uses_first_available_weekday():
    calendar = list(pd.bdate_range("2025-01-01", periods=20))

    dates = build_rebalance_dates(
        calendar,
        "2025-01-01",
        "2025-01-31",
        RebalanceConfig(frequency="weekly", weekday=0),
    )

    assert dates[0] == pd.Timestamp("2025-01-06")
    assert all(date.weekday() == 0 for date in dates)


def test_build_rebalance_dates_biweekly_takes_every_other_weekly_date():
    calendar = list(pd.bdate_range("2025-01-01", periods=30))

    weekly = build_rebalance_dates(
        calendar,
        "2025-01-01",
        "2025-02-28",
        RebalanceConfig(frequency="weekly", weekday=0),
    )
    biweekly = build_rebalance_dates(
        calendar,
        "2025-01-01",
        "2025-02-28",
        RebalanceConfig(frequency="biweekly", weekday=0),
    )

    assert biweekly == weekly[::2]


def test_build_rebalance_dates_monthly_uses_first_available_day_after_monthday():
    calendar = list(pd.bdate_range("2025-02-01", "2025-04-30"))

    dates = build_rebalance_dates(
        calendar,
        "2025-02-01",
        "2025-04-30",
        RebalanceConfig(frequency="monthly", monthday=1),
    )

    assert dates == [
        pd.Timestamp("2025-02-03"),
        pd.Timestamp("2025-03-03"),
        pd.Timestamp("2025-04-01"),
    ]


def test_select_top_candidates_returns_selected_and_full_ranking():
    candidates = [
        {"symbol": "SH603019", "score": 0.8, "rank": 1, "skip_reason": None},
        {"symbol": "SZ002241", "score": 0.5, "rank": 2, "skip_reason": None},
        {"symbol": "SH600000", "score": None, "rank": None, "skip_reason": "insufficient_history"},
    ]

    result = select_top_candidates(candidates, SelectionConfig(top_n=1))

    assert [row["symbol"] for row in result.selected] == ["SH603019"]
    assert [row["symbol"] for row in result.ranking] == ["SH603019", "SZ002241", "SH600000"]
    assert result.warnings == []


def test_select_top_candidates_keeps_skipped_rows_and_warns_when_too_few_selected():
    candidates = [
        {"symbol": "SH603019", "score": 0.8, "rank": 1, "skip_reason": None},
        {"symbol": "SZ002241", "score": None, "rank": None, "skip_reason": "insufficient_history"},
    ]

    result = select_top_candidates(candidates, SelectionConfig(top_n=2))

    assert [row["symbol"] for row in result.selected] == ["SH603019"]
    assert result.ranking[1]["skip_reason"] == "insufficient_history"
    assert result.warnings == ["selected 1 candidates, below top_n 2"]


def test_select_top_candidates_applies_score_threshold():
    candidates = [
        {"symbol": "SH603019", "score": 0.8, "rank": 1, "skip_reason": None},
        {"symbol": "SZ002241", "score": 0.2, "rank": 2, "skip_reason": None},
    ]

    result = select_top_candidates(candidates, SelectionConfig(top_n=2, score_threshold=0.5))

    assert [row["symbol"] for row in result.selected] == ["SH603019"]
    assert result.warnings == ["selected 1 candidates, below top_n 2"]
