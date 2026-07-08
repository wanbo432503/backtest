from threading import Event
import time

from portfolio_factor_optimization_models import PortfolioFactorOptimizationRequest, PortfolioFactorOptimizationResult
from portfolio_factor_optimization_progress import PortfolioFactorOptimizationJobStore
from portfolio_models import PortfolioBacktestRequest


def _request() -> PortfolioFactorOptimizationRequest:
    return PortfolioFactorOptimizationRequest(
        base_request=PortfolioBacktestRequest(
            start_date="2024-01-01",
            end_date="2026-01-01",
            universe={"mode": "auto", "symbols": [], "max_scan_symbols": 10},
            selection={"top_n": 2, "min_history_bars": 60},
        ),
        max_workers=2,
        executor_backend="thread",
    )


def test_factor_optimization_job_store_records_progress_and_result():
    progress_seen = Event()
    release_runner = Event()

    def runner(request, progress_callback=None):
        progress_callback({
            "phase": "optimizing",
            "total_trials": 4,
            "completed_trials": 1,
            "failed_trials": 0,
            "best_objective_score": 12.5,
        })
        progress_seen.set()
        release_runner.wait(timeout=2)
        return PortfolioFactorOptimizationResult(
            diagnostics={"completed_trials": 4, "max_workers": request.max_workers},
            warnings=["fixture warning"],
        )

    store = PortfolioFactorOptimizationJobStore(runner)
    created = store.submit(_request())

    assert progress_seen.wait(timeout=2)
    running = store.get(created.job_id)
    assert running.status == "running"
    assert running.phase == "optimizing"
    assert running.message == "正在并行回测候选因子"
    assert running.progress["completed_trials"] == 1
    assert running.progress["best_objective_score"] == 12.5

    release_runner.set()
    for _ in range(20):
        finished = store.get(created.job_id)
        if finished.status == "succeeded":
            break
        time.sleep(0.01)

    assert finished.status == "succeeded"
    assert finished.phase == "completed"
    assert finished.message == "因子优化完成"
    assert finished.result["diagnostics"]["completed_trials"] == 4
    assert finished.result["warnings"] == ["fixture warning"]


def test_factor_optimization_job_store_records_runner_failure():
    def runner(request, progress_callback=None):
        raise RuntimeError("optimizer boom")

    store = PortfolioFactorOptimizationJobStore(runner)
    created = store.submit(_request())

    for _ in range(20):
        snapshot = store.get(created.job_id)
        if snapshot.status == "failed":
            break
        time.sleep(0.01)

    assert snapshot.status == "failed"
    assert snapshot.phase == "failed"
    assert snapshot.message == "因子优化失败"
    assert snapshot.error == "optimizer boom"
