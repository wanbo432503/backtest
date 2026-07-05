from fastapi.testclient import TestClient

import main
from tradingagents_models import (
    TradingAgentsAnalysisResponse,
    TradingAgentsConfigResponse,
    TradingAgentsConfigTestResponse,
    TradingAgentsConfigView,
    TradingAgentsReports,
)


def test_get_tradingagents_config_masks_api_key(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_tradingagents_config_view",
        lambda: TradingAgentsConfigResponse(
            repo_path="/fake/repo",
            env_path="/fake/repo/.env",
            config=TradingAgentsConfigView(
                backend_url="http://localhost:1234/v1",
                deep_model="deep",
                quick_model="quick",
                api_key_set=True,
            ),
        ),
        raising=False,
    )
    client = TestClient(main.app)

    response = client.get("/tradingagents/config")

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["api_key_set"] is True
    assert "api_key" not in body["config"]


def test_put_tradingagents_config_saves_payload(monkeypatch):
    captured = {}

    def fake_update(payload):
        captured["payload"] = payload
        return TradingAgentsConfigResponse(
            repo_path="/fake/repo",
            env_path="/fake/repo/.env",
            config=TradingAgentsConfigView(
                backend_url=payload.backend_url,
                deep_model=payload.deep_model,
                quick_model=payload.quick_model,
            ),
        )

    monkeypatch.setattr(main, "update_tradingagents_config", fake_update, raising=False)
    client = TestClient(main.app)

    response = client.put(
        "/tradingagents/config",
        json={
            "backend_url": "http://localhost:1234/v1",
            "deep_model": "deep",
            "quick_model": "quick",
        },
    )

    assert response.status_code == 200
    assert captured["payload"].backend_url == "http://localhost:1234/v1"
    assert response.json()["config"]["deep_model"] == "deep"


def test_post_tradingagents_config_test_returns_checks(monkeypatch):
    monkeypatch.setattr(
        main,
        "test_tradingagents_config",
        lambda: TradingAgentsConfigTestResponse(
            ok=True,
            checks=[{"name": "backend_url", "ok": True}],
            warnings=[],
        ),
        raising=False,
    )
    client = TestClient(main.app)

    response = client.post("/tradingagents/config/test")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_post_tradingagents_analysis_rejects_invalid_symbol():
    client = TestClient(main.app)

    response = client.post(
        "/tradingagents/analysis",
        json={"symbol": "AAPL", "analysis_date": "2026-07-05"},
    )

    assert response.status_code == 400


def test_post_tradingagents_analysis_returns_reports(monkeypatch):
    def fake_run(payload):
        return TradingAgentsAnalysisResponse(
            symbol=payload.symbol,
            tradingagents_ticker="002241.SZ",
            analysis_date=payload.analysis_date,
            elapsed_seconds=0.1,
            reports=TradingAgentsReports(market_report="market"),
        )

    monkeypatch.setattr(main, "run_tradingagents_analysis", fake_run, raising=False)
    client = TestClient(main.app)

    response = client.post(
        "/tradingagents/analysis",
        json={"symbol": "SZ002241", "analysis_date": "2026-07-05"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tradingagents_ticker"] == "002241.SZ"
    assert body["reports"]["market_report"] == "market"
