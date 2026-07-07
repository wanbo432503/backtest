from fastapi.testclient import TestClient

import main
from portfolio_data import PortfolioDataBundle
from portfolio_models import PortfolioBacktestResult
from portfolio_progress import PortfolioJobSnapshot
from test.fixtures.portfolio_ohlcv import build_demo_portfolio_request, build_portfolio_ohlcv_fixture
from universe_scan_runner import UniverseScanResult


EXPECTED_PORTFOLIO_RESPONSE_KEYS = {
    "summary",
    "equity_curve",
    "positions",
    "trades",
    "rebalance_events",
    "candidate_rankings",
    "data_warnings",
    "risk_flags",
    "scan_diagnostics",
    "config",
}


def test_validate_universe_api_accepts_60_00_symbols():
    client = TestClient(main.app)

    response = client.post("/portfolio/validate-universe", json={"symbols": ["SH603019", "SZ002241"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["accepted_symbols"] == ["SH603019", "SZ002241"]
    assert payload["rejected"] == []


def test_validate_universe_api_rejects_unsupported_symbol():
    client = TestClient(main.app)

    response = client.post("/portfolio/validate-universe", json={"symbols": ["SH603019", "SZ300750"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["accepted_symbols"] == ["SH603019"]
    assert payload["rejected"][0]["reason"] == "unsupported_board"


def test_validate_universe_api_rejects_five_symbols():
    client = TestClient(main.app)

    response = client.post(
        "/portfolio/validate-universe",
        json={"symbols": ["SH600000", "SH601318", "SH603019", "SH605001", "SZ000001"]},
    )

    assert response.status_code == 200
    assert any(row["reason"] == "too_many_symbols" for row in response.json()["rejected"])


def test_portfolio_backtest_api_returns_runner_response(monkeypatch):
    client = TestClient(main.app)

    monkeypatch.setattr(
        main,
        "run_portfolio_backtest",
        lambda request: PortfolioBacktestResult(
            summary={"final_equity": 101000.0},
            equity_curve=[],
            positions=[],
            trades=[],
            rebalance_events=[],
            candidate_rankings=[],
            data_warnings=[],
            risk_flags=[],
            scan_diagnostics={"mode": request.universe.mode},
            config={"symbols": request.universe.symbols},
        ),
        raising=False,
    )

    response = client.post("/portfolio-backtest", json=build_demo_portfolio_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["final_equity"] == 101000.0
    assert payload["data_warnings"] == []
    assert payload["risk_flags"] == []


def test_portfolio_backtest_api_runs_engine_with_fixture_data(monkeypatch):
    client = TestClient(main.app)
    fixture = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])

    def fake_load_portfolio_ohlcv(symbols, *args, **kwargs):
        return PortfolioDataBundle(
            data_by_symbol={symbol: fixture[symbol] for symbol in symbols},
            warnings=["fixture data"],
            providers={symbol: "fixture" for symbol in symbols},
        )

    monkeypatch.setattr(
        "portfolio_backtest_runner.load_portfolio_ohlcv",
        fake_load_portfolio_ohlcv,
    )

    response = client.post("/portfolio-backtest", json=build_demo_portfolio_request())

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == EXPECTED_PORTFOLIO_RESPONSE_KEYS
    assert payload["summary"]["rebalances"] > 0
    assert payload["equity_curve"]
    assert payload["rebalance_events"]
    assert payload["candidate_rankings"]
    assert "fixture data" in payload["data_warnings"]
    assert payload["config"]["universe"]["symbols"] == ["SH603019", "SZ002241"]
    assert payload["scan_diagnostics"]["mode"] == "manual"


def test_portfolio_backtest_api_accepts_auto_universe_without_manual_symbols(monkeypatch):
    client = TestClient(main.app)
    captured = {}

    def fake_runner(request):
        captured["mode"] = request.universe.mode
        captured["symbols"] = request.universe.symbols
        return PortfolioBacktestResult(
            summary={"final_equity": 100000.0},
            equity_curve=[],
            positions=[],
            trades=[],
            rebalance_events=[],
            candidate_rankings=[],
            data_warnings=[],
            risk_flags=[],
            scan_diagnostics={"mode": request.universe.mode, "selected_count": 0},
            config=request.model_dump(mode="json"),
        )

    monkeypatch.setattr(main, "run_portfolio_backtest", fake_runner, raising=False)

    response = client.post(
        "/portfolio-backtest",
        json={
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "universe": {"mode": "auto", "symbols": [], "max_scan_symbols": 20},
            "selection": {"top_n": 2, "min_history_bars": 60},
        },
    )

    assert response.status_code == 200
    assert captured == {"mode": "auto", "symbols": []}
    assert response.json()["scan_diagnostics"]["mode"] == "auto"


def test_portfolio_universe_scan_api_returns_scan_diagnostics(monkeypatch):
    client = TestClient(main.app)

    monkeypatch.setattr(
        main,
        "run_universe_scan",
        lambda request: UniverseScanResult(
            selected_symbols=["SH603019", "SZ002241"],
            ranking=[{"symbol": "SH603019", "score": 1.0, "skip_reason": None}],
            diagnostics={
                "mode": request.universe.mode,
                "total_universe_size": 1200,
                "loaded_count": 980,
                "selected_count": 2,
            },
            warnings=["fixture warning"],
        ),
        raising=False,
    )

    response = client.post(
        "/portfolio/universe-scan",
        json={
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "universe": {"mode": "auto", "symbols": [], "max_scan_symbols": 20},
            "selection": {"top_n": 2, "min_history_bars": 60},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_symbols"] == ["SH603019", "SZ002241"]
    assert payload["scan_diagnostics"]["mode"] == "auto"
    assert payload["scan_diagnostics"]["total_universe_size"] == 1200
    assert payload["warnings"] == ["fixture warning"]


def test_portfolio_backtest_job_api_exposes_progress_status(monkeypatch):
    client = TestClient(main.app)
    snapshot = PortfolioJobSnapshot(
        job_id="job-1",
        status="running",
        phase="loading_ohlcv",
        message="正在加载行情",
        progress={
            "total_symbols": 20,
            "loaded_count": 5,
            "failed_count": 1,
        },
    )

    monkeypatch.setattr(main.portfolio_job_store, "submit", lambda request: snapshot)
    monkeypatch.setattr(main.portfolio_job_store, "get", lambda job_id: snapshot)

    create_response = client.post(
        "/portfolio-backtest/jobs",
        json={
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "universe": {"mode": "auto", "symbols": [], "max_scan_symbols": 20},
            "selection": {"top_n": 2, "min_history_bars": 60},
        },
    )

    assert create_response.status_code == 200
    assert create_response.json()["job_id"] == "job-1"

    status_response = client.get("/portfolio-backtest/jobs/job-1")
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["status"] == "running"
    assert payload["phase"] == "loading_ohlcv"
    assert payload["progress"]["loaded_count"] == 5


def test_portfolio_backtest_job_api_returns_404_for_missing_job(monkeypatch):
    client = TestClient(main.app)

    monkeypatch.setattr(main.portfolio_job_store, "get", lambda job_id: None)

    response = client.get("/portfolio-backtest/jobs/missing")

    assert response.status_code == 404


def test_portfolio_backtest_api_maps_value_error_to_400(monkeypatch):
    client = TestClient(main.app)

    def broken_runner(request):
        raise ValueError("所有组合标的数据源均获取失败")

    monkeypatch.setattr(main, "run_portfolio_backtest", broken_runner, raising=False)

    response = client.post("/portfolio-backtest", json=build_demo_portfolio_request())

    assert response.status_code == 400
    assert "所有组合标的数据源均获取失败" in response.json()["detail"]


def test_portfolio_backtest_api_maps_unexpected_error_to_500(monkeypatch):
    client = TestClient(main.app)

    def broken_runner(request):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "run_portfolio_backtest", broken_runner, raising=False)

    response = client.post("/portfolio-backtest", json=build_demo_portfolio_request())

    assert response.status_code == 500
    assert "组合回测执行失败" in response.json()["detail"]


def test_existing_backtest_endpoint_still_exists():
    routes = {route.path for route in main.app.routes}

    assert "/backtest" in routes
