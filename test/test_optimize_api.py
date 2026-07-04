from fastapi.testclient import TestClient

import main
from optimization_runner import OptimizationResult


def test_optimize_api_returns_200_for_valid_request(monkeypatch):
    client = TestClient(main.app)

    monkeypatch.setattr(
        main,
        "run_optimization",
        lambda request, strategy_registry=None: OptimizationResult(
            objective="score",
            symbols=request.optimization_config.symbols,
            top_results=[
                {
                    "rank": 1,
                    "symbol": "SH603019",
                    "strategy_name": "rsi_risk_control",
                    "params": {},
                    "train_score": 4.12,
                    "validate_score": 2.31,
                    "validate_stats": {},
                    "risk_flags": [],
                }
            ],
        ),
    )

    response = client.post(
        "/optimize",
        json={
            "start_date": "2025-07-03",
            "end_date": "2026-07-04",
            "optimization_config": {
                "symbols": ["SH603019"],
                "strategies": [{"strategy_name": "rsi_risk_control"}],
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["objective"] == "score"
    assert payload["symbols"] == ["SH603019"]
    assert payload["top_results"][0]["rank"] == 1


def test_optimize_api_rejects_non_a_share_symbol():
    client = TestClient(main.app)

    response = client.post(
        "/optimize",
        json={
            "start_date": "2025-07-03",
            "end_date": "2026-07-04",
            "optimization_config": {
                "symbols": ["AAPL"],
                "strategies": [{"strategy_name": "rsi_risk_control"}],
            },
        },
    )

    assert response.status_code == 400
    assert "仅支持 A 股代码" in response.json()["detail"]


def test_optimize_api_rejects_too_many_combinations_with_400():
    client = TestClient(main.app)

    response = client.post(
        "/optimize",
        json={
            "start_date": "2025-07-03",
            "end_date": "2026-07-04",
            "optimization_config": {
                "symbols": ["SH603019"],
                "max_combinations": 1001,
                "strategies": [{"strategy_name": "rsi_risk_control"}],
            },
        },
    )

    assert response.status_code == 400
    assert "max_combinations" in response.json()["detail"]
