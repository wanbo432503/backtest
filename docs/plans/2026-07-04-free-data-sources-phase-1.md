# Free Data Sources Phase 1.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate free A-share data sources for more accurate backtest input data and add a right-side web panel for research reports, fund flow, dragon-tiger-board activity, and announcements.

**Architecture:** Add a small data-source layer between `main.py` and external providers. Backtests call one `fetch_ohlcv()` entry point with `provider="auto"`, while market insight APIs call separate free-source helpers with rate limiting and graceful partial failure.

**Tech Stack:** FastAPI, pandas, requests/urllib, yfinance fallback, optional mootdx, Bootstrap 5, vanilla JavaScript, pytest for focused unit/API tests.

---

## Phase 1.0 Scope

Phase 1.0 includes only free or zero-key paths:

- Backtest OHLCV data:
  - A-share daily K-line: Baidu Finance first.
  - A-share realtime quote/enrichment: Tencent Finance.
  - A-share intraday/minute K-line: optional `mootdx` when installed and reachable.
  - US/HK/crypto fallback: existing `yfinance`.
  - Provider fallback: `auto -> baidu/eastmoney/mootdx where applicable -> yfinance`.
- Right-side insight panel:
  - Research reports: Eastmoney report API.
  - Fund flow: Eastmoney push2/push2his fund flow.
  - Dragon-tiger board: Eastmoney datacenter.
  - Announcements: cninfo.
- Out of scope:
  - `iwencai` semantic search because it requires an API key.
  - Full 40-endpoint import from `a-stock-data`.
  - Persistent database/cache beyond lightweight in-process rate limiting.
  - Complex corporate-action adjustment engine. Phase 1.0 must expose adjustment caveats in provider metadata.

## Phase 1.0 Subtask Map

Use this section as the canonical execution checklist. The longer task sections below contain code details and exact commands; this map defines ordering, dependencies, and done criteria.

### Subtask 1.0.1: Establish Test Baseline

**Purpose:** Make the intended contracts executable before touching implementation.

**Depends on:** None.

**Outputs:**
- `requirements.txt` includes `pytest`.
- `test/test_market_data_sources.py` covers symbol normalization and Baidu payload parsing.
- `test/test_market_insights_api.py` covers the new insights endpoint contract.

**Done when:**
- New tests fail for the expected reasons before implementation.
- Existing project tests still import cleanly after adding tests.

### Subtask 1.0.2: Build Symbol and Provider Core

**Purpose:** Centralize market detection, ticker normalization, OHLCV cleanup, and provider result metadata.

**Depends on:** 1.0.1.

**Outputs:**
- New `market_data.py`.
- `detect_market()`, `normalize_symbol()`, `prepare_ohlcv()`, `DataSourceResult`.
- `fetch_yfinance_ohlcv()` retained as the fallback provider.

**Done when:**
- CN/US/HK/crypto normalization tests pass.
- `prepare_ohlcv()` returns `Open`, `High`, `Low`, `Close`, `Volume` with a sorted `DatetimeIndex`.

### Subtask 1.0.3: Add Free A-Share OHLCV Providers

**Purpose:** Improve A-share backtest data precision without paid APIs.

**Depends on:** 1.0.2.

**Outputs:**
- Baidu daily K-line parser and fetcher.
- Optional `mootdx` fetcher for intraday/minute A-share data.
- `fetch_ohlcv(..., provider="auto")` selector.

**Done when:**
- Baidu parser unit test passes.
- `provider="auto"` chooses Baidu for CN daily symbols.
- `provider="auto"` still falls back to `yfinance` for AAPL, `0700.HK`, and `BTC-USD`.
- `mootdx` failure is non-fatal when the package is not installed or TCP is unreachable.

### Subtask 1.0.4: Wire Backtest API to Provider Layer

**Purpose:** Replace the hard-coded `yfinance` path in `/backtest` with selectable data providers.

**Depends on:** 1.0.3.

**Outputs:**
- `BacktestRequest.data_provider`.
- `/backtest` calls `fetch_ohlcv()`.
- Response includes `data_provider` and `data_warnings`.
- Web form includes a data source selector.

**Done when:**
- Existing backtest behavior remains compatible when `data_provider` is omitted.
- Frontend submits `data_provider`.
- Returned provider metadata appears in the stats area.

### Subtask 1.0.5: Add Market Insights Service

**Purpose:** Collect A-share contextual data for the new right-side panel.

**Depends on:** 1.0.2.

**Outputs:**
- New `market_insights.py`.
- Eastmoney `em_get()` with serial throttling.
- Report, fund-flow, dragon-tiger-board, and cninfo announcement loaders.
- `get_market_insights(symbol)` aggregator.

**Done when:**
- A failure in one upstream source produces a warning, not a 500 for the whole aggregation.
- Non-CN symbols return empty sections with a clear warning.
- The unit/API test can monkeypatch `get_market_insights()` and verify response shape.

### Subtask 1.0.6: Add Market Insights API Endpoint

**Purpose:** Expose the insights service to the web UI.

**Depends on:** 1.0.5.

**Outputs:**
- `GET /market-insights/{symbol}` in `main.py`.
- JSON response with `reports`, `fund_flow`, `dragon_tiger`, `announcements`, and `warnings`.

**Done when:**
- `python -m pytest test/test_market_insights_api.py -v` passes.
- Invalid or empty symbols return a 400.
- Unexpected aggregation errors return a clear 500 message.

### Subtask 1.0.7: Add Right-Side Insights Panel

**Purpose:** Show research reports, fund flow, dragon-tiger-board records, and announcements next to backtest output.

**Depends on:** 1.0.6.

**Outputs:**
- `templates/index.html` uses a three-column desktop layout.
- Right panel contains four compact sections.
- JavaScript fetches `/market-insights/{symbol}` after symbol selection and after successful backtest.

**Done when:**
- Wide screens show left controls, center chart, right insights.
- Smaller screens stack the insights panel below without overlap.
- Empty/error sections render gracefully.

### Subtask 1.0.8: Documentation and Verification

**Purpose:** Make the new data behavior understandable and prove the main path works.

**Depends on:** 1.0.4, 1.0.7.

**Outputs:**
- README section documenting Phase 1.0 data sources and limitations.
- Focused tests pass.
- One live Baidu smoke test is recorded in terminal output.

**Done when:**
- `python -m pytest test/test_market_data_sources.py test/test_market_insights_api.py -v` passes.
- `python - <<'PY' ... import main ... PY` prints `ok`.
- README clearly states that `iwencai` is excluded because it requires an API key.

## Detailed TodoList

### Setup and Baseline

- [x] Confirm working tree state with `git status --short`.
- [x] Add `pytest` to `requirements.txt`.
- [x] Create `test/test_market_data_sources.py`.
- [x] Add tests for `detect_market()` with `600519`, `SH600519`, `600519.SH`, `000001.SZ`.
- [x] Add tests for `normalize_symbol()` plain A-share code output.
- [x] Add tests proving `AAPL`, `0700.HK`, and `BTC-USD` remain yfinance-compatible.
- [x] Create `test/test_market_insights_api.py`.
- [x] Add API shape test for `/market-insights/600519`.
- [x] Run `python -m pytest test/test_market_data_sources.py -v`.
- [x] Confirm expected failure is missing `market_data`.
- [x] Run `python -m pytest test/test_market_insights_api.py -v`.
- [x] Confirm expected failure is missing endpoint or insight function.

### Market Data Core

- [x] Create `market_data.py`.
- [x] Add `NormalizedSymbol` dataclass.
- [x] Add `DataSourceResult` dataclass.
- [x] Implement `detect_market(symbol)`.
- [x] Implement `normalize_symbol(symbol)`.
- [x] Implement `prepare_ohlcv(data)`.
- [x] Implement `fetch_yfinance_ohlcv(symbol, start_date, end_date, interval)`.
- [x] Run normalization tests.
- [x] Fix normalization until tests pass.
- [x] Commit `market_data.py` and normalization tests.

### A-Share Provider Integration

- [x] Add Baidu sample payload parser test.
- [x] Implement `parse_baidu_kline_payload(payload)`.
- [x] Implement `fetch_baidu_daily_ohlcv(symbol, start_date, end_date)`.
- [x] Verify Baidu parser converts rows into `Open`, `High`, `Low`, `Close`, `Volume`, `Amount`.
- [x] Add optional `fetch_mootdx_ohlcv(symbol, interval)` implementation.
- [x] Ensure missing `mootdx` raises `ValueError`, not `ImportError`.
- [x] Implement `fetch_ohlcv(symbol, start_date, end_date, interval="1d", provider="auto")`.
- [x] Ensure `fetch_ohlcv()` records provider-specific warnings.
- [x] Run `python -m pytest test/test_market_data_sources.py -v`.
- [x] Commit A-share providers.

### Backtest API Wiring

- [x] Import `fetch_ohlcv` in `main.py`.
- [x] Add `data_provider: str = "auto"` to `BacktestRequest`.
- [x] Replace direct `yf.Ticker(...).history(...)` block with `fetch_ohlcv(...)`.
- [x] Keep existing date validation and strategy validation.
- [x] Keep existing minimum 50-row validation.
- [x] Add `data_provider` to `/backtest` response.
- [x] Add `data_warnings` to `/backtest` response.
- [x] Add data source selector to `templates/index.html`.
- [x] Include `data_provider` in submitted `formData`.
- [x] Render returned provider metadata in the stats area.
- [x] Run app import smoke test.
- [x] Commit backtest provider wiring.

### Market Insights Backend

- [x] Create `market_insights.py`.
- [x] Add shared `UA`, `EM_SESSION`, `EM_MIN_INTERVAL`, and `_em_last_call`.
- [x] Implement `em_get()` with serial throttling and jitter.
- [x] Implement `_secid(code)`.
- [x] Implement `fetch_eastmoney_reports(code)`.
- [x] Implement `fetch_fund_flow_120d(code)`.
- [x] Implement `fetch_dragon_tiger_board(code)`.
- [x] Implement `fetch_cninfo_announcements(code)`.
- [x] Implement `get_market_insights(symbol)`.
- [x] Ensure each section loader failure is captured in `warnings`.
- [x] Ensure non-CN symbols return empty sections and one warning.
- [x] Import `get_market_insights` in `main.py`.
- [x] Add `GET /market-insights/{symbol}`.
- [x] Run `python -m pytest test/test_market_insights_api.py -v`.
- [x] Commit insight backend.

### Right-Side Web Panel

- [x] Change main layout from two columns to responsive three columns.
- [x] Preserve left control form behavior.
- [x] Preserve center chart and stats behavior.
- [x] Add right-side card titled `标的信息`.
- [x] Add panel status element.
- [x] Add warning element.
- [x] Add `研报` list container.
- [x] Add `资金` list container.
- [x] Add `龙虎榜` list container.
- [x] Add `公告` list container.
- [x] Add `.insight-section`, `.insight-item`, and `.insight-empty` CSS.
- [x] Add `renderInsightList(containerId, items, formatter)`.
- [x] Add `loadMarketInsights(symbol)`.
- [x] Call `loadMarketInsights(symbol)` from `setSymbol(symbol)`.
- [x] Call `loadMarketInsights(formData.symbol)` after successful backtest.
- [x] Verify empty/error sections do not break page layout.
- [x] Commit right-side panel.

### Verification and Docs

- [x] Add README section for `数据源 Phase 1.0`.
- [x] Document free sources: Baidu, Tencent, optional mootdx, Eastmoney, cninfo, yfinance fallback.
- [x] Document excluded source: `iwencai` because it requires API key.
- [x] Document provider caveats: mootdx unadjusted prices, Eastmoney rate limiting, free interface instability.
- [x] If `main.py` import fails because `static/` is missing, create `static/.gitkeep`.
- [x] Run `python -m pytest test/test_market_data_sources.py test/test_market_insights_api.py -v`.
- [x] Run `python - <<'PY'` import smoke for `main`, `market_data`, and `market_insights`.
- [x] Run one live Baidu `fetch_ohlcv("600519", ..., "baidu")` smoke test and record current `ResultCode=403` provider-side failure.
- [x] Run one live `fetch_ohlcv("600519", ..., "auto")` fallback smoke test and verify it returns rows through yfinance A-share symbol conversion.
- [x] Start `python main.py`.
- [x] Open `http://localhost:8005`.
- [x] Run one A-share daily backtest with provider `auto`.
- [x] Verify provider metadata appears in stats.
- [x] Verify right-side panel loads all four sections or clear warnings.
- [x] Commit documentation and final verification fixes.

## Implementation Order

1. Execute Subtask 1.0.1 first and commit tests.
2. Execute Subtask 1.0.2 and commit core provider abstractions.
3. Execute Subtask 1.0.3 and commit free A-share OHLCV providers.
4. Execute Subtask 1.0.4 and commit backtest API/UI provider selector.
5. Execute Subtask 1.0.5 and 1.0.6 together only if the endpoint test remains small.
6. Execute Subtask 1.0.7 and manually check layout before committing.
7. Execute Subtask 1.0.8 last and record any provider caveats in README.

## Phase 1.0 Non-Goals

- [x] Do not implement paid API support.
- [x] Do not require `IWENCAI_API_KEY`.
- [x] Do not add a database.
- [x] Do not add complex backtest strategy changes.
- [x] Do not rewrite the frontend framework.
- [x] Do not import the whole `a-stock-data` `SKILL.md` wholesale.
- [x] Do not make Eastmoney calls concurrently.
- [x] Do not make `mootdx` mandatory for app startup.

## Task 1: Add Test Harness

**Files:**
- Modify: `requirements.txt`
- Create: `test/test_market_data_sources.py`
- Create: `test/test_market_insights_api.py`

**Step 1: Add pytest dependency**

Append:

```txt
pytest
```

**Step 2: Add failing tests for symbol normalization**

Create `test/test_market_data_sources.py`:

```python
import pandas as pd

from market_data import normalize_symbol, detect_market


def test_detects_a_share_formats():
    assert detect_market("600519") == "CN"
    assert detect_market("SH600519") == "CN"
    assert detect_market("600519.SH") == "CN"
    assert detect_market("000001.SZ") == "CN"


def test_normalizes_a_share_to_plain_code():
    assert normalize_symbol("SH600519").code == "600519"
    assert normalize_symbol("600519.SH").code == "600519"
    assert normalize_symbol("SZ000001").code == "000001"


def test_non_cn_symbols_remain_yfinance_compatible():
    assert normalize_symbol("AAPL").symbol == "AAPL"
    assert normalize_symbol("0700.HK").symbol == "0700.HK"
    assert normalize_symbol("BTC-USD").symbol == "BTC-USD"
```

Run:

```bash
python -m pytest test/test_market_data_sources.py -v
```

Expected: FAIL because `market_data.py` does not exist yet.

**Step 3: Add failing API test for insight endpoint**

Create `test/test_market_insights_api.py`:

```python
from fastapi.testclient import TestClient

import main


def test_market_insights_endpoint_returns_sections(monkeypatch):
    monkeypatch.setattr(main, "get_market_insights", lambda symbol: {
        "symbol": symbol,
        "reports": [{"title": "Report A"}],
        "fund_flow": [{"date": "2026-07-03", "main_net_inflow": 123.0}],
        "dragon_tiger": [],
        "announcements": [{"title": "Announcement A"}],
        "warnings": [],
    })

    client = TestClient(main.app)
    response = client.get("/market-insights/600519")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "600519"
    assert set(["reports", "fund_flow", "dragon_tiger", "announcements", "warnings"]).issubset(body)
```

Run:

```bash
python -m pytest test/test_market_insights_api.py -v
```

Expected: FAIL because `get_market_insights` and `/market-insights/{symbol}` do not exist.

**Step 4: Commit after passing later**

```bash
git add requirements.txt test/test_market_data_sources.py test/test_market_insights_api.py
git commit -m "test: cover phase 1 data source contracts"
```

## Task 2: Create Market Data Abstraction

**Files:**
- Create: `market_data.py`
- Modify: `main.py`
- Test: `test/test_market_data_sources.py`

**Step 1: Implement normalization and provider result types**

Create `market_data.py`:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class NormalizedSymbol:
    raw: str
    symbol: str
    code: str
    market: str
    exchange_prefix: str = ""


@dataclass
class DataSourceResult:
    data: pd.DataFrame
    provider: str
    warnings: list[str]


def detect_market(symbol: str) -> str:
    value = symbol.strip().upper()
    if value.endswith(".HK"):
        return "HK"
    if value.startswith(("SH", "SZ", "BJ")):
        return "CN"
    if value.endswith((".SH", ".SZ", ".BJ")):
        return "CN"
    if value.isdigit() and len(value) == 6:
        return "CN"
    return "US"


def normalize_symbol(symbol: str) -> NormalizedSymbol:
    raw = symbol.strip()
    value = raw.upper()
    market = detect_market(value)
    if market != "CN":
        return NormalizedSymbol(raw=raw, symbol=value, code=value, market=market)

    code = value
    if code.startswith(("SH", "SZ", "BJ")):
        code = code[2:]
    if code.endswith((".SH", ".SZ", ".BJ")):
        code = code.split(".")[0]

    if code.startswith(("6", "9")):
        prefix = "sh"
    elif code.startswith("8"):
        prefix = "bj"
    else:
        prefix = "sz"

    return NormalizedSymbol(raw=raw, symbol=f"{prefix}{code}", code=code, market="CN", exchange_prefix=prefix)


def prepare_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(col).title() for col in data.columns]
    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"数据缺少必要的列: {missing}")
    if "Volume" not in data.columns:
        data["Volume"] = 0
    for col in required + ["Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=required)
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    return data.sort_index()
```

**Step 2: Add yfinance provider**

Append:

```python
def fetch_yfinance_ohlcv(symbol: str, start_date: str, end_date: str, interval: str) -> DataSourceResult:
    data = yf.Ticker(symbol).history(start=start_date, end=end_date, interval=interval)
    if data.empty:
        raise ValueError("yfinance returned empty data")
    return DataSourceResult(data=prepare_ohlcv(data), provider="yfinance", warnings=[])
```

**Step 3: Run normalization tests**

```bash
python -m pytest test/test_market_data_sources.py::test_detects_a_share_formats test/test_market_data_sources.py::test_normalizes_a_share_to_plain_code -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add market_data.py test/test_market_data_sources.py
git commit -m "feat: add market data abstraction"
```

## Task 3: Add Free A-Share OHLCV Providers

**Files:**
- Modify: `market_data.py`
- Test: `test/test_market_data_sources.py`

**Step 1: Add Baidu daily K-line parser test**

Append to `test/test_market_data_sources.py`:

```python
from market_data import parse_baidu_kline_payload


def test_parses_baidu_kline_payload():
    payload = {
        "ResultCode": "0",
        "Result": {
            "newMarketData": {
                "keys": ["timestamp", "time", "open", "close", "volume", "high", "low", "amount"],
                "marketData": "1783353600,2026-07-03,1205.24,1194.45,34268,1210.14,1185.00,4099266243"
            }
        }
    }

    frame = parse_baidu_kline_payload(payload)

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume", "Amount"]
    assert frame.iloc[0]["Close"] == 1194.45
```

Run:

```bash
python -m pytest test/test_market_data_sources.py::test_parses_baidu_kline_payload -v
```

Expected: FAIL because parser does not exist.

**Step 2: Implement Baidu K-line provider**

Append to `market_data.py`:

```python
import requests


def parse_baidu_kline_payload(payload: dict) -> pd.DataFrame:
    if str(payload.get("ResultCode", "-1")) != "0":
        raise ValueError(f"百度股市通返回错误: {payload.get('ResultCode')}")
    market_data = payload.get("Result", {}).get("newMarketData", {})
    keys = market_data.get("keys", [])
    rows = [row for row in market_data.get("marketData", "").split(";") if row.strip()]
    records = []
    for row in rows:
        values = row.split(",")
        item = dict(zip(keys, values))
        records.append({
            "Date": item.get("time"),
            "Open": item.get("open"),
            "High": item.get("high"),
            "Low": item.get("low"),
            "Close": item.get("close"),
            "Volume": item.get("volume", 0),
            "Amount": item.get("amount", 0),
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("百度股市通返回空 K 线")
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.set_index("Date")
    return prepare_ohlcv(frame)


def fetch_baidu_daily_ohlcv(symbol: NormalizedSymbol, start_date: str, end_date: str) -> DataSourceResult:
    url = "https://finance.pae.baidu.com/selfselect/getstockquotation"
    params = {
        "all": "1",
        "isIndex": "false",
        "isBk": "false",
        "isBlock": "false",
        "isFutures": "false",
        "isStock": "true",
        "newFormat": "1",
        "group": "quotation_kline_ab",
        "finClientType": "pc",
        "code": symbol.code,
        "start_time": start_date.replace("-", ""),
        "ktype": "1",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.finance-web.v1+json",
        "Origin": "https://gushitong.baidu.com",
        "Referer": "https://gushitong.baidu.com/",
    }
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    frame = parse_baidu_kline_payload(response.json())
    end = pd.to_datetime(end_date)
    frame = frame[(frame.index >= pd.to_datetime(start_date)) & (frame.index <= end)]
    if frame.empty:
        raise ValueError("百度股市通在指定时间区间无数据")
    return DataSourceResult(
        data=frame,
        provider="baidu",
        warnings=["百度股市通日 K 线用于 A 股免费回测数据源；复权口径需后续单独校验。"],
    )
```

**Step 3: Add optional mootdx provider**

Append:

```python
def fetch_mootdx_ohlcv(symbol: NormalizedSymbol, interval: str) -> DataSourceResult:
    try:
        from mootdx.quotes import Quotes
    except ImportError as exc:
        raise ValueError("mootdx is not installed") from exc

    frequency_map = {
        "1m": 8,
        "5m": 0,
        "15m": 1,
        "30m": 2,
        "60m": 3,
        "1h": 3,
        "1d": 9,
        "1wk": 5,
        "1mo": 6,
    }
    frequency = frequency_map.get(interval)
    if frequency is None:
        raise ValueError(f"mootdx 不支持该频率: {interval}")

    client = Quotes.factory(market="std")
    data = client.bars(symbol=symbol.code, frequency=frequency, offset=800)
    if data is None or data.empty:
        raise ValueError("mootdx returned empty data")
    data = data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "vol": "Volume"})
    data = data.set_index(pd.to_datetime(data["datetime"]))
    return DataSourceResult(
        data=prepare_ohlcv(data),
        provider="mootdx",
        warnings=["mootdx bars 返回不复权原始价，跨除权除息日回测需谨慎。"],
    )
```

**Step 4: Add auto provider selector**

Append:

```python
def fetch_ohlcv(symbol: str, start_date: str, end_date: str, interval: str = "1d", provider: str = "auto") -> DataSourceResult:
    normalized = normalize_symbol(symbol)
    attempts = []

    if provider in ("auto", "baidu") and normalized.market == "CN" and interval == "1d":
        attempts.append(lambda: fetch_baidu_daily_ohlcv(normalized, start_date, end_date))
    if provider in ("auto", "mootdx") and normalized.market == "CN":
        attempts.append(lambda: fetch_mootdx_ohlcv(normalized, interval))
    if provider in ("auto", "yfinance"):
        attempts.append(lambda: fetch_yfinance_ohlcv(symbol, start_date, end_date, interval))

    errors = []
    for attempt in attempts:
        try:
            result = attempt()
            if not result.data.empty:
                return result
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError("所有数据源均获取失败: " + " | ".join(errors))
```

**Step 5: Run tests**

```bash
python -m pytest test/test_market_data_sources.py -v
```

Expected: PASS.

**Step 6: Commit**

```bash
git add market_data.py test/test_market_data_sources.py
git commit -m "feat: add free a-share ohlcv providers"
```

## Task 4: Route Backtests Through Provider Layer

**Files:**
- Modify: `main.py`
- Modify: `templates/index.html`
- Test: `test/test_market_insights_api.py`

**Step 1: Extend request model**

In `main.py`, import provider:

```python
from market_data import fetch_ohlcv
```

Add field to `BacktestRequest`:

```python
data_provider: str = "auto"
```

**Step 2: Replace yfinance-only history fetch**

Replace the block around existing `yf.Ticker(...).history(...)` with:

```python
source_result = fetch_ohlcv(
    request.symbol,
    request.start_date,
    request.end_date,
    request.interval,
    request.data_provider,
)
data = source_result.data
```

Keep the existing `prepare_data_for_backtesting(data)` call for compatibility, or replace it with a thin wrapper around `market_data.prepare_ohlcv()` in a later cleanup.

**Step 3: Include provider metadata in response**

Add response fields:

```python
"data_provider": source_result.provider,
"data_warnings": source_result.warnings,
```

**Step 4: Add data-provider selector to form**

In `templates/index.html`, under interval selector, add:

```html
<div class="mb-3">
    <label for="dataProvider" class="form-label">数据源</label>
    <select class="form-select" id="dataProvider" required>
        <option value="auto" selected>自动选择</option>
        <option value="baidu">百度股市通（日线A股）</option>
        <option value="mootdx">通达信 mootdx（A股）</option>
        <option value="yfinance">Yahoo Finance</option>
    </select>
</div>
```

Add to `formData`:

```javascript
data_provider: document.getElementById('dataProvider').value
```

Display provider result near stats:

```javascript
if (result.data_provider) {
    const providerDiv = document.createElement('div');
    providerDiv.className = 'alert alert-info small mb-2';
    providerDiv.textContent = `数据源: ${result.data_provider}`;
    statsContent.prepend(providerDiv);
}
```

**Step 5: Run app import test**

```bash
python - <<'PY'
import main
print("ok")
PY
```

Expected: `ok`.

**Step 6: Commit**

```bash
git add main.py templates/index.html
git commit -m "feat: use selectable market data providers"
```

## Task 5: Add Market Insights Service

**Files:**
- Create: `market_insights.py`
- Modify: `main.py`
- Test: `test/test_market_insights_api.py`

**Step 1: Implement rate-limited Eastmoney helper and insight shell**

Create `market_insights.py`:

```python
import random
import time
from datetime import datetime, timedelta

import requests

from market_data import normalize_symbol

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
EM_SESSION = requests.Session()
EM_SESSION.headers.update({"User-Agent": UA})
EM_MIN_INTERVAL = 1.0
_em_last_call = [0.0]


def em_get(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 15):
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
            "reports": [],
            "fund_flow": [],
            "dragon_tiger": [],
            "announcements": [],
            "warnings": ["Phase 1.0 insight panel only supports A-share symbols."],
        }

    sections = {}
    for key, loader in {
        "reports": fetch_eastmoney_reports,
        "fund_flow": fetch_fund_flow_120d,
        "dragon_tiger": fetch_dragon_tiger_board,
        "announcements": fetch_cninfo_announcements,
    }.items():
        try:
            sections[key] = loader(normalized.code)
        except Exception as exc:
            sections[key] = []
            warnings.append(f"{key} 获取失败: {exc}")

    return {"symbol": normalized.code, **sections, "warnings": warnings}
```

**Step 2: Add reports loader**

Append:

```python
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
    rows = data.get("data") or []
    return [{
        "title": row.get("title") or row.get("TITLE") or "",
        "org": row.get("orgSName") or row.get("ORG_S_NAME") or "",
        "date": row.get("publishDate") or row.get("PUBLISH_DATE") or "",
        "rating": row.get("emRatingName") or row.get("EM_RATING_NAME") or "",
    } for row in rows[:page_size]]
```

**Step 3: Add fund flow loader**

Append:

```python
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
        rows.append({
            "date": values[0],
            "main_net_inflow": float(values[1]) if len(values) > 1 and values[1] else 0.0,
            "small_net_inflow": float(values[5]) if len(values) > 5 and values[5] else 0.0,
        })
    return rows
```

**Step 4: Add dragon-tiger and announcements loaders**

Append:

```python
def fetch_dragon_tiger_board(code: str, look_back: int = 30) -> list[dict]:
    end = datetime.now()
    start = end - timedelta(days=look_back)
    params = {
        "reportName": "RPT_DAILYBILLBOARD_DETAILS",
        "columns": "ALL",
        "filter": f"(SECURITY_CODE=\"{code}\")(TRADE_DATE>='{start:%Y-%m-%d}')(TRADE_DATE<='{end:%Y-%m-%d}')",
        "pageNumber": "1",
        "pageSize": "10",
        "source": "WEB",
        "client": "WEB",
    }
    response = em_get("https://datacenter-web.eastmoney.com/api/data/v1/get", params=params, timeout=15)
    response.raise_for_status()
    rows = response.json().get("result", {}).get("data", []) or []
    return [{
        "date": row.get("TRADE_DATE", ""),
        "reason": row.get("EXPLAIN", ""),
        "net_buy": row.get("NET_BUY_AMT", ""),
    } for row in rows[:10]]


def fetch_cninfo_announcements(code: str, page_size: int = 5) -> list[dict]:
    payload = {
        "stock": code,
        "tabName": "fulltext",
        "pageSize": str(page_size),
        "pageNum": "1",
        "column": "szse",
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
    response = requests.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=payload, headers=headers, timeout=15)
    response.raise_for_status()
    rows = response.json().get("announcements", []) or []
    return [{
        "title": row.get("announcementTitle", ""),
        "date": row.get("announcementTime", ""),
        "url": "https://www.cninfo.com.cn/new/disclosure/detail?annoId=" + str(row.get("announcementId", "")),
    } for row in rows[:page_size]]
```

**Step 5: Wire FastAPI endpoint**

In `main.py`, import:

```python
from market_insights import get_market_insights
```

Add endpoint:

```python
@app.get("/market-insights/{symbol}")
async def market_insights_endpoint(symbol: str):
    if not symbol or not symbol.strip():
        raise HTTPException(status_code=400, detail="股票代码不能为空")
    try:
        return get_market_insights(symbol.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取市场信息失败: {exc}")
```

**Step 6: Run tests**

```bash
python -m pytest test/test_market_insights_api.py -v
```

Expected: PASS.

**Step 7: Commit**

```bash
git add market_insights.py main.py test/test_market_insights_api.py
git commit -m "feat: add market insights endpoint"
```

## Task 6: Add Right-Side Web Panel

**Files:**
- Modify: `templates/index.html`

**Step 1: Change layout to three columns**

Change the main grid:

```html
<div class="col-xl-3 col-lg-4">
```

for the left controls, then:

```html
<div class="col-xl-6 col-lg-8">
```

for the chart area, and add a new right column after it:

```html
<div class="col-xl-3 col-lg-12">
    <div class="card">
        <div class="card-header">
            <i class="fas fa-newspaper"></i> 标的信息
        </div>
        <div class="card-body">
            <div id="insightStatus" class="small text-muted">选择标的后加载</div>
            <div id="insightWarnings" class="alert alert-warning small mt-2" style="display:none;"></div>
            <div class="insight-section">
                <h6>研报</h6>
                <div id="reportsList" class="small"></div>
            </div>
            <div class="insight-section">
                <h6>资金</h6>
                <div id="fundFlowList" class="small"></div>
            </div>
            <div class="insight-section">
                <h6>龙虎榜</h6>
                <div id="dragonTigerList" class="small"></div>
            </div>
            <div class="insight-section">
                <h6>公告</h6>
                <div id="announcementsList" class="small"></div>
            </div>
        </div>
    </div>
</div>
```

**Step 2: Add compact panel CSS**

In `<style>`:

```css
.insight-section {
    border-top: 1px solid #e9ecef;
    padding-top: 10px;
    margin-top: 10px;
}
.insight-section h6 {
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.insight-item {
    padding: 6px 0;
    border-bottom: 1px solid #f1f3f5;
}
.insight-item:last-child {
    border-bottom: none;
}
.insight-empty {
    color: #6c757d;
}
```

**Step 3: Add panel JavaScript**

Add functions:

```javascript
function renderInsightList(containerId, items, formatter) {
    const container = document.getElementById(containerId);
    if (!items || items.length === 0) {
        container.innerHTML = '<div class="insight-empty">暂无数据</div>';
        return;
    }
    container.innerHTML = items.map(formatter).join('');
}

async function loadMarketInsights(symbol) {
    const status = document.getElementById('insightStatus');
    const warnings = document.getElementById('insightWarnings');
    status.textContent = '正在加载...';
    warnings.style.display = 'none';

    try {
        const response = await fetch(`/market-insights/${encodeURIComponent(symbol)}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '加载失败');
        }
        const data = await response.json();
        status.textContent = `标的: ${data.symbol}`;
        if (data.warnings && data.warnings.length) {
            warnings.style.display = 'block';
            warnings.textContent = data.warnings.join('；');
        }
        renderInsightList('reportsList', data.reports, item =>
            `<div class="insight-item"><strong>${item.title || '-'}</strong><br><span class="text-muted">${item.org || ''} ${item.date || ''}</span></div>`);
        renderInsightList('fundFlowList', data.fund_flow, item =>
            `<div class="insight-item">${item.date || '-'} 主力净流入: ${item.main_net_inflow || 0}</div>`);
        renderInsightList('dragonTigerList', data.dragon_tiger, item =>
            `<div class="insight-item">${item.date || '-'} ${item.reason || ''}<br><span class="text-muted">净买入: ${item.net_buy || '-'}</span></div>`);
        renderInsightList('announcementsList', data.announcements, item =>
            `<div class="insight-item"><a href="${item.url || '#'}" target="_blank">${item.title || '-'}</a><br><span class="text-muted">${item.date || ''}</span></div>`);
    } catch (error) {
        status.textContent = `加载失败: ${error.message}`;
    }
}
```

Call after successful backtest:

```javascript
loadMarketInsights(formData.symbol);
```

Also call in `setSymbol(symbol)` after updating the input:

```javascript
loadMarketInsights(symbol);
```

**Step 4: Manual browser check**

Run:

```bash
python main.py
```

Open `http://localhost:8005`, choose `600519`, run backtest, and verify the right panel loads without overlapping the chart.

**Step 5: Commit**

```bash
git add templates/index.html
git commit -m "feat: add market insights side panel"
```

## Task 7: Verification and Documentation

**Files:**
- Modify: `README.md`
- Optional create: `static/.gitkeep` if FastAPI static mount errors in the local environment.

**Step 1: Document data sources and limitations**

Add a README section:

```markdown
## 数据源 Phase 1.0

回测数据通过 `provider=auto` 自动选择：

- A股日线优先使用百度股市通免费 K 线。
- A股分钟线可选使用 mootdx；未安装或网络不可达时回退。
- 美股、港股、加密货币继续使用 yfinance。
- 研报、资金流、龙虎榜来自东财公开接口，已串行限流。
- 公告来自巨潮 cninfo。

注意：mootdx K 线是不复权原始价；东财接口有风控，批量调用需降低频率。
```

**Step 2: Run focused tests**

```bash
python -m pytest test/test_market_data_sources.py test/test_market_insights_api.py -v
```

Expected: PASS.

**Step 3: Run smoke import**

```bash
python - <<'PY'
import main
import market_data
import market_insights
print("ok")
PY
```

Expected: `ok`.

**Step 4: Run one live API smoke test**

```bash
python - <<'PY'
from market_data import fetch_ohlcv
result = fetch_ohlcv("600519", "2026-06-01", "2026-07-04", "1d", "baidu")
print(result.provider, len(result.data), result.data.tail(1).to_dict("records"))
PY
```

Expected: provider is `baidu`, data length > 0.

**Step 5: Commit**

```bash
git add README.md static/.gitkeep
git commit -m "docs: document phase 1 data sources"
```

## Acceptance Criteria

- `/backtest` accepts `data_provider` and returns `data_provider` plus `data_warnings`.
- A-share daily backtests can run without yfinance when Baidu returns data.
- US/HK/crypto symbols keep working through yfinance.
- `/market-insights/{symbol}` returns the four panel sections and never fails the whole response because one upstream source is unavailable.
- The right-side web panel displays reports, fund flow, dragon-tiger-board records, and announcements in a responsive third column.
- Tests pass with `python -m pytest test/test_market_data_sources.py test/test_market_insights_api.py -v`.

## Risks and Mitigations

- **Free interfaces can change without notice:** isolate each provider in `market_data.py` or `market_insights.py` so failures are contained.
- **Eastmoney rate limiting:** all Eastmoney requests must go through `em_get()` with serial throttling.
- **mootdx network instability:** treat mootdx as optional, never as the only provider.
- **A-share corporate actions:** expose warnings in Phase 1.0; plan a Phase 1.1 adjustment module after live validation.
- **Frontend density:** use a three-column layout only on wide screens; stack the insight panel below on smaller screens.
