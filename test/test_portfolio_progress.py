from threading import Event
import time

from portfolio_models import PortfolioBacktestRequest, PortfolioBacktestResult
from portfolio_progress import PortfolioBacktestJobStore


def test_portfolio_job_store_records_progress_and_result():
    progress_seen = Event()
    release_runner = Event()

    def runner(request, progress_callback=None):
        progress_callback({
            "phase": "loading_ohlcv",
            "total_symbols": 3,
            "loaded_count": 1,
            "failed_count": 0,
            "current_symbol": "SH600000",
        })
        progress_seen.set()
        release_runner.wait(timeout=2)
        return PortfolioBacktestResult(
            summary={"final_equity": 101000},
            scan_diagnostics={"mode": request.universe.mode},
        )

    store = PortfolioBacktestJobStore(runner)
    request = PortfolioBacktestRequest(
        start_date="2025-01-01",
        end_date="2025-12-31",
        universe={"mode": "auto", "symbols": [], "max_scan_symbols": 3},
    )

    created = store.submit(request)
    assert progress_seen.wait(timeout=2)
    running = store.get(created.job_id)
    assert running.status == "running"
    assert running.phase == "loading_ohlcv"
    assert running.progress["current_symbol"] == "SH600000"

    release_runner.set()
    for _ in range(20):
        finished = store.get(created.job_id)
        if finished.status == "succeeded":
            break
        time.sleep(0.01)

    assert finished.status == "succeeded"
    assert finished.result["summary"]["final_equity"] == 101000


def test_portfolio_job_store_records_runner_failure():
    def runner(request, progress_callback=None):
        raise RuntimeError("boom")

    store = PortfolioBacktestJobStore(runner)
    request = PortfolioBacktestRequest(start_date="2025-01-01", end_date="2025-12-31")

    created = store.submit(request)
    for _ in range(20):
        snapshot = store.get(created.job_id)
        if snapshot.status == "failed":
            break
        time.sleep(0.01)

    assert snapshot.status == "failed"
    assert snapshot.phase == "failed"
    assert snapshot.error == "boom"
