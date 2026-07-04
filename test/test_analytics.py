import math

import pandas as pd

from analytics import calculate_score, extract_core_metrics, parse_percent


def test_calculate_score_uses_phase_2_formula():
    score = calculate_score(cagr_pct=17.36, sharpe=0.51, max_drawdown_pct=20.15)

    assert score == 0.4 * 17.36 + 0.3 * 0.51 - 0.3 * 20.15


def test_extract_core_metrics_normalizes_negative_drawdown_and_adds_score():
    stats = {
        "Return (Ann.) [%]": "17.36%",
        "Sharpe Ratio": 0.51,
        "Max. Drawdown [%]": "-20.15%",
        "# Trades": 8,
    }

    metrics = extract_core_metrics(stats)

    assert metrics["annual_return_pct"] == 17.36
    assert metrics["sharpe"] == 0.51
    assert metrics["max_drawdown_pct"] == 20.15
    assert metrics["score"] == calculate_score(17.36, 0.51, 20.15)


def test_extract_core_metrics_treats_nan_sharpe_as_zero():
    stats = pd.Series(
        {
            "Return (Ann.) [%]": 10,
            "Sharpe Ratio": math.nan,
            "Max. Drawdown [%]": -5,
            "# Trades": 4,
        }
    )

    metrics = extract_core_metrics(stats)

    assert metrics["sharpe"] == 0
    assert metrics["score"] == calculate_score(10, 0, 5)


def test_extract_core_metrics_applies_min_trades_filter_and_risk_flag():
    stats = {
        "Return (Ann.) [%]": 12,
        "Sharpe Ratio": 1.2,
        "Max. Drawdown [%]": -35,
        "# Trades": 3,
    }

    metrics = extract_core_metrics(stats, min_trades=5, max_drawdown_limit_pct=30)

    assert metrics["trades"] == 3
    assert metrics["is_rankable"] is False
    assert metrics["is_high_risk"] is True
    assert "交易次数不足" in metrics["risk_notes"]
    assert "最大回撤超过阈值" in metrics["risk_notes"]


def test_parse_percent_accepts_numeric_and_percent_strings():
    assert parse_percent("17.36%") == 17.36
    assert parse_percent("-20.15%") == -20.15
    assert parse_percent(8) == 8
