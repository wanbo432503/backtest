from __future__ import annotations

import math

import pandas as pd


RAW_PRICE_COLUMNS = {
    "Open": "RawOpen",
    "High": "RawHigh",
    "Low": "RawLow",
    "Close": "RawClose",
}
ACTION_COLUMNS = {
    "fenhong": "CashDividendPer10",
    "songzhuangu": "BonusSharesPer10",
    "peigu": "RightsSharesPer10",
    "peigujia": "RightsPrice",
}


def apply_corporate_actions(
    data: pd.DataFrame,
    actions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return dynamic-forward-adjusted OHLC plus untouched execution prices."""
    frame = data.copy().sort_index()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    if frame.index.has_duplicates:
        raise ValueError("日线行情包含重复日期，无法计算除权因子")
    if frame.empty:
        return frame

    for adjusted_name, raw_name in RAW_PRICE_COLUMNS.items():
        source_name = raw_name if raw_name in frame.columns else adjusted_name
        frame[raw_name] = pd.to_numeric(frame[source_name], errors="coerce")
    frame["AdjFactor"] = 1.0
    for output_name in ACTION_COLUMNS.values():
        frame[output_name] = 0.0

    normalized_actions = normalize_corporate_actions(actions)
    if not normalized_actions.empty:
        normalized_actions = normalized_actions[
            (normalized_actions.index >= frame.index.min())
            & (normalized_actions.index <= frame.index.max())
        ]

    factors = pd.Series(1.0, index=frame.index, dtype="float64")
    action_schedule = []
    for action_date, action in normalized_actions.iterrows():
        locations = frame.index.searchsorted(action_date, side="left")
        if locations >= len(frame.index):
            continue
        effective_date = frame.index[locations]
        for source_name, output_name in ACTION_COLUMNS.items():
            frame.loc[effective_date, output_name] += float(action[source_name])
        reference_price = None
        if locations == 0:
            action_schedule.append(
                _action_schedule_record(action_date, action, reference_price)
            )
            continue

        previous_close = float(frame.iloc[locations - 1]["RawClose"])
        cash_dividend = float(action["fenhong"])
        bonus_shares = float(action["songzhuangu"])
        rights_shares = float(action["peigu"])
        rights_price = float(action["peigujia"])
        denominator = 10.0 + bonus_shares + rights_shares
        reference_price = (
            previous_close * 10.0
            - cash_dividend
            + rights_shares * rights_price
        ) / denominator
        event_factor = reference_price / previous_close
        if (
            not math.isfinite(event_factor)
            or event_factor <= 0
            or event_factor > 2
        ):
            raise ValueError(
                f"{effective_date.date()} 除权因子无效: {event_factor!r}"
            )
        factors.iloc[:locations] *= event_factor
        action_schedule.append(
            _action_schedule_record(action_date, action, reference_price)
        )

    frame["AdjFactor"] = factors
    for adjusted_name, raw_name in RAW_PRICE_COLUMNS.items():
        frame[adjusted_name] = frame[raw_name] * frame["AdjFactor"]
    frame.attrs["corporate_actions"] = action_schedule
    return frame


def _action_schedule_record(
    action_date: pd.Timestamp,
    action: pd.Series,
    reference_price: float | None,
) -> dict[str, float | str | None]:
    return {
        "date": pd.Timestamp(action_date).strftime("%Y-%m-%d"),
        **{
            output_name: float(action[source_name])
            for source_name, output_name in ACTION_COLUMNS.items()
        },
        "RawReferencePrice": reference_price,
    }


def normalize_corporate_actions(actions: pd.DataFrame | None) -> pd.DataFrame:
    columns = list(ACTION_COLUMNS)
    if actions is None or actions.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    normalized = actions.copy()
    if not isinstance(normalized.index, pd.DatetimeIndex):
        if {"year", "month", "day"}.issubset(normalized.columns):
            normalized.index = pd.to_datetime(
                normalized[["year", "month", "day"]].rename(
                    columns={"year": "year", "month": "month", "day": "day"}
                )
            )
        elif "date" in normalized.columns:
            normalized.index = pd.to_datetime(normalized["date"])
        else:
            raise ValueError("除权数据缺少日期")
    normalized.index = pd.DatetimeIndex(normalized.index).normalize()
    if "category" in normalized.columns:
        normalized = normalized[
            pd.to_numeric(normalized["category"], errors="coerce") == 1
        ]
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = 0.0
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)
    if normalized.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    rows = []
    for action_date, group in normalized.groupby(level=0, sort=True):
        rights_shares = float(group["peigu"].sum())
        rights_cost = float((group["peigu"] * group["peigujia"]).sum())
        rows.append(
            {
                "date": action_date,
                "fenhong": float(group["fenhong"].sum()),
                "songzhuangu": float(group["songzhuangu"].sum()),
                "peigu": rights_shares,
                "peigujia": rights_cost / rights_shares if rights_shares else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("date")[columns]


def detect_unexplained_discontinuities(
    data: pd.DataFrame,
    *,
    threshold_pct: float = 22.0,
) -> list[dict[str, float | str]]:
    if data.empty or "Close" not in data.columns:
        return []
    closes = pd.to_numeric(data["Close"], errors="coerce")
    returns = closes.pct_change()
    rows = []
    for date, value in returns[returns.abs() > threshold_pct / 100].items():
        rows.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "change_pct": round(float(value) * 100, 6),
            }
        )
    return rows
