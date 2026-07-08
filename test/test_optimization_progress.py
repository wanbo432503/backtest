from threading import Event
import time

from optimization_models import OptimizationConfig, OptimizationRequest, StrategyParamConfig
from optimization_progress import OptimizationJobStore
from optimization_runner import OptimizationResult


def _request() -> OptimizationRequest:
    return OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[StrategyParamConfig(strategy_name="rsi_risk_control")],
        ),
    )


def test_optimization_job_store_records_progress_and_result():
    progress_seen = Event()
    release_runner = Event()

    def runner(request, progress_callback=None, strategy_registry=None):
        progress_callback({
            "phase": "optimizing",
            "total_trials": 4,
            "completed_trials": 1,
            "current_symbol": "SH603019",
            "current_strategy": "rsi_risk_control",
        })
        progress_seen.set()
        release_runner.wait(timeout=2)
        return OptimizationResult(
            objective="score",
            symbols=request.optimization_config.symbols,
            top_results=[{"rank": 1, "validate_score": 2.5}],
        )

    store = OptimizationJobStore(runner, strategy_registry={"rsi_risk_control": object})
    created = store.submit(_request())

    assert progress_seen.wait(timeout=2)
    running = store.get(created.job_id)
    assert running.status == "running"
    assert running.phase == "optimizing"
    assert running.message == "正在回测参数候选"
    assert running.progress["completed_trials"] == 1
    assert running.progress["current_symbol"] == "SH603019"

    release_runner.set()
    for _ in range(20):
        finished = store.get(created.job_id)
        if finished.status == "succeeded":
            break
        time.sleep(0.01)

    assert finished.status == "succeeded"
    assert finished.phase == "completed"
    assert finished.message == "参数优化完成"
    assert finished.result["top_results"][0]["rank"] == 1


def test_optimization_job_store_records_runner_failure():
    def runner(request, progress_callback=None, strategy_registry=None):
        raise RuntimeError("optimizer boom")

    store = OptimizationJobStore(runner)
    created = store.submit(_request())

    for _ in range(20):
        snapshot = store.get(created.job_id)
        if snapshot.status == "failed":
            break
        time.sleep(0.01)

    assert snapshot.status == "failed"
    assert snapshot.phase == "failed"
    assert snapshot.message == "参数优化失败"
    assert snapshot.error == "optimizer boom"
