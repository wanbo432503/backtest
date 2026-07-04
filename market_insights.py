from __future__ import annotations

import random
import time
from datetime import datetime, timedelta

import requests

from market_data import fetch_tencent_quote, normalize_symbol


UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
EM_SESSION = requests.Session()
EM_SESSION.headers.update({"User-Agent": UA})
EM_MIN_INTERVAL = 1.0
_em_last_call = [0.0]


def em_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 15,
):
    wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))
    try:
        return EM_SESSION.get(url, params=params, headers=headers, timeout=timeout)
    finally:
        _em_last_call[0] = time.time()


def get_market_insights(symbol: str) -> dict:
    normalized = normalize_symbol(symbol)
    warnings = []

    if normalized.market != "CN":
        return {
            "symbol": symbol,
            "quote": {},
            "reports": [],
            "fund_flow": [],
            "dragon_tiger": [],
            "announcements": [],
            "warnings": ["Phase 1.0 标的信息面板仅支持 A 股。"],
        }

    sections = {}
    loaders = {
        "quote": lambda code: fetch_tencent_quote([code]).get(code, {}),
        "reports": fetch_eastmoney_reports,
        "fund_flow": fetch_fund_flow_120d,
        "dragon_tiger": fetch_dragon_tiger_board,
        "announcements": fetch_cninfo_announcements,
    }

    for key, loader in loaders.items():
        try:
            sections[key] = loader(normalized.code)
        except Exception as exc:
            sections[key] = {} if key == "quote" else []
            warnings.append(f"{key} 获取失败: {exc}")

    return {"symbol": normalized.code, **sections, "warnings": warnings}


def fetch_eastmoney_reports(code: str, page_size: int = 5) -> list[dict]:
    params = {
        "industryCode": "*",
        "pageSize": str(page_size),
        "industry": "*",
        "rating": "*",
        "ratingChange": "*",
        "beginTime": "2000-01-01",
        "endTime": "2030-01-01",
        "pageNo": "1",
        "qType": "0",
        "code": code,
        "source": "WEB",
        "client": "WEB",
    }
    response = em_get(
        "https://reportapi.eastmoney.com/report/list",
        params=params,
        headers={"Referer": "https://data.eastmoney.com/"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    rows = data.get("data") or data.get("result", {}).get("data") or []
    return [
        {
            "title": row.get("title") or row.get("TITLE") or "",
            "org": row.get("orgSName") or row.get("ORG_S_NAME") or "",
            "date": row.get("publishDate") or row.get("PUBLISH_DATE") or "",
            "rating": row.get("emRatingName") or row.get("EM_RATING_NAME") or "",
        }
        for row in rows[:page_size]
    ]


def _secid(code: str) -> str:
    prefix = "1" if code.startswith(("6", "9")) else "0"
    return f"{prefix}.{code}"


def fetch_fund_flow_120d(code: str, limit: int = 10) -> list[dict]:
    params = {
        "lmt": "120",
        "klt": "101",
        "secid": _secid(code),
        "fields1": "f1,f2,f3",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
    }
    response = em_get(
        "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
        params=params,
        headers={"Referer": "https://quote.eastmoney.com/"},
        timeout=15,
    )
    response.raise_for_status()
    klines = response.json().get("data", {}).get("klines", []) or []
    rows = []
    for line in klines[-limit:]:
        values = line.split(",")
        rows.append(
            {
                "date": _value_at(values, 0),
                "main_net_inflow": _float_at(values, 1),
                "small_net_inflow": _float_at(values, 5),
            }
        )
    return rows


def fetch_dragon_tiger_board(code: str, look_back: int = 30) -> list[dict]:
    end = datetime.now()
    start = end - timedelta(days=look_back)
    params = {
        "reportName": "RPT_DAILYBILLBOARD_DETAILS",
        "columns": "ALL",
        "filter": f'(SECURITY_CODE="{code}")(TRADE_DATE>=\'{start:%Y-%m-%d}\')(TRADE_DATE<=\'{end:%Y-%m-%d}\')',
        "pageNumber": "1",
        "pageSize": "10",
        "source": "WEB",
        "client": "WEB",
    }
    response = em_get("https://datacenter-web.eastmoney.com/api/data/v1/get", params=params, timeout=15)
    response.raise_for_status()
    result = response.json().get("result") or {}
    rows = result.get("data", []) or []
    return [
        {
            "date": row.get("TRADE_DATE", ""),
            "reason": row.get("EXPLAIN", ""),
            "net_buy": row.get("NET_BUY_AMT", ""),
        }
        for row in rows[:10]
    ]


def fetch_cninfo_announcements(code: str, page_size: int = 5) -> list[dict]:
    column = "sse" if code.startswith(("6", "9")) else "szse"
    payload = {
        "stock": code,
        "tabName": "fulltext",
        "pageSize": str(page_size),
        "pageNum": "1",
        "column": column,
        "category": "",
        "plate": "",
        "seDate": "",
        "searchkey": "",
        "secid": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    headers = {
        "User-Agent": UA,
        "Referer": "https://www.cninfo.com.cn/new/disclosure",
        "Origin": "https://www.cninfo.com.cn",
    }
    response = requests.post(
        "https://www.cninfo.com.cn/new/hisAnnouncement/query",
        data=payload,
        headers=headers,
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json().get("announcements", []) or []
    return [
        {
            "title": row.get("announcementTitle", ""),
            "date": row.get("announcementTime", ""),
            "url": "https://www.cninfo.com.cn/new/disclosure/detail?annoId="
            + str(row.get("announcementId", "")),
        }
        for row in rows[:page_size]
    ]


def _value_at(values: list[str], index: int) -> str:
    return values[index] if len(values) > index else ""


def _float_at(values: list[str], index: int) -> float:
    try:
        return float(_value_at(values, index))
    except (TypeError, ValueError):
        return 0.0
