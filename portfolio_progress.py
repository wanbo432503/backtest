from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Literal

from portfolio_models import PortfolioBacktestRequest, PortfolioBacktestResult


JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class PortfolioJobSnapshot:
    job_id: str
    status: JobStatus
    phase: str = "queued"
    message: str = "等待执行"
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())

    def to_api_response(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class PortfolioBacktestJobStore:
    def __init__(
        self,
        runner: Callable[..., PortfolioBacktestResult],
        *,
        max_workers: int = 1,
        job_label: str = "组合回测",
    ) -> None:
        self._runner = runner
        self._job_label = job_label
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, PortfolioJobSnapshot] = {}
        self._lock = Lock()

    def submit(self, request: PortfolioBacktestRequest) -> PortfolioJobSnapshot:
        job_id = uuid.uuid4().hex
        snapshot = PortfolioJobSnapshot(job_id=job_id, status="queued")
        with self._lock:
            self._jobs[job_id] = snapshot
        self._executor.submit(self._run_job, job_id, request)
        return snapshot

    def get(self, job_id: str) -> PortfolioJobSnapshot | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(
        self,
        job_id: str,
        *,
        phase: str,
        message: str | None = None,
        progress: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            snapshot = self._jobs.get(job_id)
            if snapshot is None:
                return
            snapshot.status = "running"
            snapshot.phase = phase
            snapshot.message = message or _message_for_phase(phase)
            snapshot.progress = progress or {}
            snapshot.updated_at = _utc_now()

    def _run_job(self, job_id: str, request: PortfolioBacktestRequest) -> None:
        self.update_progress(job_id, phase="starting", message=f"开始{self._job_label}")
        try:
            result = self._runner(
                request,
                progress_callback=lambda event: self.update_progress(
                    job_id,
                    phase=str(event.get("phase") or "running"),
                    progress={key: value for key, value in event.items() if key != "phase"},
                ),
            )
            with self._lock:
                snapshot = self._jobs[job_id]
                snapshot.status = "succeeded"
                snapshot.phase = "completed"
                snapshot.message = f"{self._job_label}完成"
                snapshot.result = result.to_api_response()
                snapshot.error = None
                snapshot.updated_at = _utc_now()
        except Exception as exc:
            with self._lock:
                snapshot = self._jobs[job_id]
                snapshot.status = "failed"
                snapshot.phase = "failed"
                snapshot.message = f"{self._job_label}失败"
                snapshot.error = str(exc)
                snapshot.updated_at = _utc_now()


def _message_for_phase(phase: str) -> str:
    labels = {
        "discovering_universe": "正在发现 60/00 股票池",
        "loading_ohlcv": "正在加载行情数据",
        "loading_fundamentals": "正在加载财务因子",
        "scoring": "正在计算候选评分",
        "backtesting": "正在执行组合调仓回测",
        "building_signals": "正在计算多股票交易信号",
        "signal_backtesting": "正在执行多股票信号组合回测",
        "completed": "组合回测完成",
    }
    return labels.get(phase, "正在执行")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
