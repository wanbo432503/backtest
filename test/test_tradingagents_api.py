from fastapi.testclient import TestClient

import main
from tradingagents_models import (
    TradingAgentsAnalysisResponse,
    TradingAgentsConfigResponse,
    TradingAgentsConfigTestResponse,
    TradingAgentsConfigView,
    TradingAgentsPortfolioSummaryResponse,
    TradingAgentsReports,
)


def test_get_tradingagents_config_masks_api_key(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_tradingagents_config_view",
        lambda: TradingAgentsConfigResponse(
            repo_path="/fake/backtest",
            env_path="/fake/backtest/.env",
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


def test_get_tradingagents_config_api_key_returns_secret(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_tradingagents_config_api_key",
        lambda: {"api_key": "secret-value", "api_key_set": True},
        raising=False,
    )
    client = TestClient(main.app)

    response = client.get("/tradingagents/config/api-key")

    assert response.status_code == 200
    assert response.json() == {"api_key": "secret-value", "api_key_set": True}


def test_put_tradingagents_config_saves_payload(monkeypatch):
    captured = {}

    def fake_update(payload):
        captured["payload"] = payload
        return TradingAgentsConfigResponse(
            repo_path="/fake/backtest",
            env_path="/fake/backtest/.env",
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


def test_post_tradingagents_portfolio_summary_returns_explanation(monkeypatch):
    captured = {}

    def fake_summary(payload):
        captured["payload"] = payload
        return TradingAgentsPortfolioSummaryResponse(
            summary_text="组合总结",
            elapsed_seconds=0.1,
            warnings=["AI summary is explanatory only and is not used by backtest metrics."],
        )

    monkeypatch.setattr(main, "run_tradingagents_portfolio_summary", fake_summary, raising=False)
    client = TestClient(main.app)

    response = client.post(
        "/tradingagents/portfolio-summary",
        json={
            "selected_symbols": ["SH603019", "SZ002241"],
            "summary_metrics": {"final_equity": 101000},
            "latest_candidate_rankings": [{"symbol": "SH603019", "score": 0.9}],
            "risk_flags": ["high_drawdown"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary_text"] == "组合总结"
    assert "api_key" not in body
    assert captured["payload"].selected_symbols == ["SH603019", "SZ002241"]


def test_post_tradingagents_portfolio_summary_accepts_twenty_symbols(monkeypatch):
    captured = {}

    def fake_summary(payload):
        captured["payload"] = payload
        return TradingAgentsPortfolioSummaryResponse(summary_text="组合总结", elapsed_seconds=0.1)

    monkeypatch.setattr(main, "run_tradingagents_portfolio_summary", fake_summary, raising=False)
    client = TestClient(main.app)
    selected_symbols = [f"SH6000{index:02d}" for index in range(1, 21)]

    response = client.post(
        "/tradingagents/portfolio-summary",
        json={
            "selected_symbols": selected_symbols,
            "summary_metrics": {"final_equity": 101000},
            "latest_candidate_rankings": [],
            "risk_flags": [],
        },
    )

    assert response.status_code == 200
    assert captured["payload"].selected_symbols == selected_symbols


def test_post_tradingagents_portfolio_summary_rejects_invalid_symbol():
    client = TestClient(main.app)

    response = client.post(
        "/tradingagents/portfolio-summary",
        json={
            "selected_symbols": ["SZ300750"],
            "summary_metrics": {},
            "latest_candidate_rankings": [],
            "risk_flags": [],
        },
    )

    assert response.status_code == 400
