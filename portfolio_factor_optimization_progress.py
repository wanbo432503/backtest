from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Literal

from portfolio_factor_optimization_models import (
    PortfolioFactorOptimizationRequest,
    PortfolioFactorOptimizationResult,
)


OptimizationJobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class PortfolioFactorOptimizationJobSnapshot:
    job_id: str
    status: OptimizationJobStatus
    phase: str = "queued"
    message: str = "等待因子优化"
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


class PortfolioFactorOptimizationJobStore:
    def __init__(
        self,
        runner: Callable[..., PortfolioFactorOptimizationResult],
        *,
        max_jobs: int = 1,
    ) -> None:
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=max_jobs)
        self._jobs: dict[str, PortfolioFactorOptimizationJobSnapshot] = {}
        self._lock = Lock()

    def submit(
        self,
        request: PortfolioFactorOptimizationRequest,
    ) -> PortfolioFactorOptimizationJobSnapshot:
        job_id = uuid.uuid4().hex
        snapshot = PortfolioFactorOptimizationJobSnapshot(job_id=job_id, status="queued")
        with self._lock:
            self._jobs[job_id] = snapshot
        self._executor.submit(self._run_job, job_id, request)
        return snapshot

    def get(self, job_id: str) -> PortfolioFactorOptimizationJobSnapshot | None:
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

    def _run_job(
        self,
        job_id: str,
        request: PortfolioFactorOptimizationRequest,
    ) -> None:
        self.update_progress(job_id, phase="starting", message="开始因子优化")
        try:
            result = self._runner(
                request,
                progress_callback=lambda event: self.update_progress(
                    job_id,
                    phase=str(event.get("phase") or "running"),
                    message=event.get("message"),
                    progress={
                        key: value
                        for key, value in event.items()
                        if key not in {"phase", "message"}
                    },
                ),
            )
            with self._lock:
                snapshot = self._jobs[job_id]
                snapshot.status = "succeeded"
                snapshot.phase = "completed"
                snapshot.message = "因子优化完成"
                snapshot.result = result.model_dump(mode="json")
                snapshot.error = None
                snapshot.updated_at = _utc_now()
        except Exception as exc:
            with self._lock:
                snapshot = self._jobs[job_id]
                snapshot.status = "failed"
                snapshot.phase = "failed"
                snapshot.message = "因子优化失败"
                snapshot.error = str(exc)
                snapshot.updated_at = _utc_now()


def _message_for_phase(phase: str) -> str:
    labels = {
        "starting": "开始因子优化",
        "loading_context": "正在加载股票池行情",
        "optimizing": "正在并行回测候选因子",
        "completed": "因子优化完成",
    }
    return labels.get(phase, "正在执行因子优化")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
