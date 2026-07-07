from fastapi.testclient import TestClient

import main
from portfolio_models import PortfolioBacktestResult
from test.fixtures.portfolio_ohlcv import build_demo_portfolio_request


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
