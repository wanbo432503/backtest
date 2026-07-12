from fastapi.testclient import TestClient

import main
from optimization_runner import OptimizationResult


def test_optimize_api_returns_200_for_valid_request(monkeypatch):
    client = TestClient(main.app)

    monkeypatch.setattr(
        main,
        "run_optimization",
        lambda request, strategy_library=None: OptimizationResult(
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


def test_optimize_api_rejects_multiple_symbols():
    client = TestClient(main.app)

    response = client.post(
        "/optimize",
        json={
            "start_date": "2025-07-03",
            "end_date": "2026-07-04",
            "optimization_config": {
                "symbols": ["SH603019", "SZ002241"],
                "strategies": [{"strategy_name": "rsi_risk_control"}],
            },
        },
    )

    assert response.status_code == 400
    assert "symbols" in response.json()["detail"]


def test_optimize_api_accepts_user_defined_combinations_above_1000(monkeypatch):
    client = TestClient(main.app)

    monkeypatch.setattr(
        main,
        "run_optimization",
        lambda request, strategy_library=None: OptimizationResult(
            objective="score",
            symbols=request.optimization_config.symbols,
            top_results=[],
            progress_log=[f"max_combinations={request.optimization_config.max_combinations}"],
        ),
    )

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

    assert response.status_code == 200
    assert response.json()["progress_log"] == ["max_combinations=1001"]


def test_optimization_job_api_creates_and_reads_job(monkeypatch):
    client = TestClient(main.app)

    class FakeSnapshot:
        job_id = "job-1"

        def to_api_response(self):
            return {
                "job_id": self.job_id,
                "status": "running",
                "phase": "optimizing",
                "message": "正在回测参数候选",
                "progress": {"total_trials": 3, "completed_trials": 1},
                "result": None,
                "error": None,
            }

    class FakeStore:
        snapshot = FakeSnapshot()

        def submit(self, request):
            assert request.optimization_config.symbols == ["SH603019"]
            return self.snapshot

        def get(self, job_id):
            assert job_id == "job-1"
            return self.snapshot

    monkeypatch.setattr(main, "optimization_job_store", FakeStore())

    request_payload = {
        "start_date": "2025-07-03",
        "end_date": "2026-07-04",
        "optimization_config": {
            "symbols": ["SH603019"],
            "strategies": [{"strategy_name": "rsi_risk_control"}],
        },
    }
    created = client.post("/optimization/jobs", json=request_payload)
    fetched = client.get("/optimization/jobs/job-1")

    assert created.status_code == 200
    assert created.json()["progress"]["total_trials"] == 3
    assert fetched.status_code == 200
    assert fetched.json()["job_id"] == "job-1"
