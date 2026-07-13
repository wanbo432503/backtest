from __future__ import annotations

import pandas as pd
import pytest

from price_adjustment import apply_corporate_actions, detect_unexplained_discontinuities


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [52.5, 35.28, 34.0],
            "High": [53.8, 35.8, 34.8],
            "Low": [52.0, 33.0, 33.5],
            "Close": [53.05, 33.8, 34.2],
            "Volume": [1_000, 2_000, 1_500],
        },
        index=pd.to_datetime(["2015-06-18", "2015-06-19", "2015-06-22"]),
    )


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": [1],
            "fenhong": [0.8],
            "songzhuangu": [5.0],
            "peigu": [0.0],
            "peigujia": [0.0],
        },
        index=pd.to_datetime(["2015-06-19"]),
    )


def test_adjustment_preserves_raw_prices_and_removes_ex_right_gap():
    adjusted = apply_corporate_actions(_raw_frame(), _actions())

    expected_factor = ((53.05 * 10 - 0.8) / 15) / 53.05
    assert adjusted.loc["2015-06-18", "RawClose"] == 53.05
    assert adjusted.loc["2015-06-19", "RawClose"] == 33.8
    assert adjusted.loc["2015-06-18", "AdjFactor"] == pytest.approx(expected_factor)
    assert adjusted.loc["2015-06-19", "AdjFactor"] == 1
    assert adjusted.loc["2015-06-18", "Close"] == pytest.approx(35.3133333333)
    expected_return = 33.8 / ((53.05 * 10 - 0.8) / 15) - 1
    assert adjusted.loc["2015-06-19", "Close"] / adjusted.loc["2015-06-18", "Close"] - 1 == pytest.approx(
        expected_return
    )
    assert adjusted.loc["2015-06-19", "CashDividendPer10"] == 0.8
    assert adjusted.loc["2015-06-19", "BonusSharesPer10"] == 5


def test_future_action_after_frame_is_not_used_for_adjustment():
    actions = pd.concat(
        [
            _actions(),
            pd.DataFrame(
                {
                    "category": [1],
                    "fenhong": [10.0],
                    "songzhuangu": [10.0],
                    "peigu": [0.0],
                    "peigujia": [0.0],
                },
                index=pd.to_datetime(["2016-06-19"]),
            ),
        ]
    )

    adjusted = apply_corporate_actions(_raw_frame(), actions)

    assert adjusted.loc["2015-06-19", "AdjFactor"] == 1
    assert adjusted.loc["2015-06-22", "AdjFactor"] == 1


def test_unexplained_discontinuity_uses_adjusted_not_raw_returns():
    raw = _raw_frame()
    adjusted = apply_corporate_actions(raw, _actions())

    assert len(detect_unexplained_discontinuities(raw, threshold_pct=20)) == 1
    assert detect_unexplained_discontinuities(adjusted, threshold_pct=20) == []


def test_invalid_corporate_action_factor_fails_loudly():
    actions = _actions()
    actions.loc[:, "fenhong"] = 600

    with pytest.raises(ValueError, match="除权因子"):
        apply_corporate_actions(_raw_frame(), actions)
