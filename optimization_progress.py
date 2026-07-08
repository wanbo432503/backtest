from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Literal

from optimization_models import OptimizationRequest
from optimization_runner import OptimizationResult


OptimizationJobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class OptimizationJobSnapshot:
    job_id: str
    status: OptimizationJobStatus
    phase: str = "queued"
    message: str = "等待参数优化"
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


class OptimizationJobStore:
    def __init__(
        self,
        runner: Callable[..., OptimizationResult],
        *,
        strategy_registry: dict[str, Any] | None = None,
        max_jobs: int = 1,
    ) -> None:
        self._runner = runner
        self._strategy_registry = strategy_registry
        self._executor = ThreadPoolExecutor(max_workers=max_jobs)
        self._jobs: dict[str, OptimizationJobSnapshot] = {}
        self._lock = Lock()

    def submit(self, request: OptimizationRequest) -> OptimizationJobSnapshot:
        job_id = uuid.uuid4().hex
        snapshot = OptimizationJobSnapshot(job_id=job_id, status="queued")
        with self._lock:
            self._jobs[job_id] = snapshot
        self._executor.submit(self._run_job, job_id, request)
        return snapshot

    def get(self, job_id: str) -> OptimizationJobSnapshot | None:
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

    def _run_job(self, job_id: str, request: OptimizationRequest) -> None:
        self.update_progress(job_id, phase="starting", message="开始参数优化")
        try:
            result = self._runner(
                request,
                strategy_registry=self._strategy_registry,
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
                snapshot.message = "参数优化完成"
                snapshot.result = result.to_api_response()
                snapshot.error = None
                snapshot.updated_at = _utc_now()
        except Exception as exc:
            with self._lock:
                snapshot = self._jobs[job_id]
                snapshot.status = "failed"
                snapshot.phase = "failed"
                snapshot.message = "参数优化失败"
                snapshot.error = str(exc)
                snapshot.updated_at = _utc_now()


def _message_for_phase(phase: str) -> str:
    labels = {
        "starting": "开始参数优化",
        "preparing": "正在生成参数候选",
        "optimizing": "正在回测参数候选",
        "completed": "参数优化完成",
    }
    return labels.get(phase, "正在执行参数优化")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
