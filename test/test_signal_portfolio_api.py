from fastapi.testclient import TestClient

import main
from portfolio_progress import PortfolioJobSnapshot


def _payload():
    return {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "universe": {"mode": "manual", "symbols": ["SH603019", "SZ002241"]},
        "risk": {"max_positions": 2},
    }


def test_strategy_catalog_exposes_same_nine_strategies_to_signal_portfolios():
    client = TestClient(main.app)

    response = client.get("/strategies")

    assert response.status_code == 200
    catalog = response.json()
    assert len(catalog) == 9
    assert all(item["engine"] == "unified" for item in catalog)
    assert all("signal_portfolio" in item["supported_modes"] for item in catalog)
    assert all(item["class_name"] for item in catalog)


def test_signal_portfolio_job_api_creates_and_reads_job(monkeypatch):
    client = TestClient(main.app)
    snapshot = PortfolioJobSnapshot(
        job_id="signal-job-1",
        status="running",
        phase="signal_backtesting",
        progress={"completed_days": 10, "total_days": 100},
    )
    monkeypatch.setattr(main.signal_portfolio_job_store, "submit", lambda request: snapshot)
    monkeypatch.setattr(main.signal_portfolio_job_store, "get", lambda job_id: snapshot)

    created = client.post("/signal-portfolio-backtest/jobs", json=_payload())
    fetched = client.get("/signal-portfolio-backtest/jobs/signal-job-1")

    assert created.status_code == 200
    assert created.json()["job_id"] == "signal-job-1"
    assert fetched.status_code == 200
    assert fetched.json()["progress"]["completed_days"] == 10


def test_signal_portfolio_api_normalizes_nested_strategy_before_submit(monkeypatch):
    client = TestClient(main.app)
    snapshot = PortfolioJobSnapshot(job_id="normalized-1", status="queued")
    submitted = []
    monkeypatch.setattr(
        main.signal_portfolio_job_store,
        "submit",
        lambda request: submitted.append(request) or snapshot,
    )
    payload = _payload()
    payload["strategy"] = {
        "strategy_name": "rsi_risk_control",
        "parameters": {"rsi_period": 8},
    }

    response = client.post("/signal-portfolio-backtest/jobs", json=payload)

    assert response.status_code == 200
    assert submitted[0].strategy.parameters["rsi_period"] == 8
    assert "rsi_buy" in submitted[0].strategy.parameters


def test_signal_portfolio_api_rejects_unknown_strategy_before_submit(monkeypatch):
    client = TestClient(main.app)
    submitted = []
    monkeypatch.setattr(
        main.signal_portfolio_job_store,
        "submit",
        lambda request: submitted.append(request),
    )
    payload = _payload()
    payload["strategy"] = {"strategy_name": "missing", "parameters": {}}

    response = client.post("/signal-portfolio-backtest/jobs", json=payload)

    assert response.status_code == 400
    assert submitted == []


def test_signal_portfolio_api_rejects_unknown_strategy_parameter(monkeypatch):
    client = TestClient(main.app)
    submitted = []
    monkeypatch.setattr(
        main.signal_portfolio_job_store,
        "submit",
        lambda request: submitted.append(request),
    )
    payload = _payload()
    payload["strategy"] = {
        "strategy_name": "rsi_risk_control",
        "parameters": {"unknown_parameter": 1},
    }

    response = client.post("/signal-portfolio-backtest/jobs", json=payload)

    assert response.status_code == 400
    assert submitted == []


def test_signal_portfolio_auto_mode_ignores_fixed_pool_symbols(monkeypatch):
    client = TestClient(main.app)
    snapshot = PortfolioJobSnapshot(job_id="signal-auto-1", status="queued")
    submitted = []
    monkeypatch.setattr(
        main.signal_portfolio_job_store,
        "submit",
        lambda request: submitted.append(request) or snapshot,
    )
    payload = _payload()
    payload["universe"] = {
        "mode": "auto",
        "symbols": ["SH603019", "SZ002241"],
        "max_scan_symbols": 3000,
    }

    response = client.post("/signal-portfolio-backtest/jobs", json=payload)

    assert response.status_code == 200
    assert submitted[0].universe.mode == "auto"
    assert submitted[0].universe.symbols == []
    assert submitted[0].universe.max_scan_symbols == 3000


def test_signal_portfolio_job_api_rejects_invalid_manual_pool():
    client = TestClient(main.app)
    payload = _payload()
    payload["universe"] = {"mode": "manual", "symbols": ["AAPL"]}

    response = client.post("/signal-portfolio-backtest/jobs", json=payload)

    assert response.status_code == 400


def test_signal_portfolio_job_api_returns_404_for_missing_job(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main.signal_portfolio_job_store, "get", lambda job_id: None)

    response = client.get("/signal-portfolio-backtest/jobs/missing")

    assert response.status_code == 404
