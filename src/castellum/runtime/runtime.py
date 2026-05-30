from __future__ import annotations

import asyncio
import uuid
from typing import Any, TYPE_CHECKING

from castellum.core.context import PipelineContext, set_current_context
from castellum.metrics.collector import MetricsCollector
from castellum.runtime.retry import RetryPolicy
from castellum.runtime.scheduler import Scheduler

if TYPE_CHECKING:
    from castellum.core.pipeline import Pipeline


class Runtime:
    def __init__(
        self,
        *,
        max_cpu_workers: int = 4,
        max_gpu_workers: int = 1,
        max_concurrent_remote_calls: int = 10,
        tokens_per_minute_limit: int | None = None,
        remote_limits: dict[str, dict[str, int]] | None = None,
        default_timeout_seconds: float = 60.0,
        retry_policy: dict[str, Any] | None = None,
        metrics_backend: str = "none",
        metrics_namespace: str = "castellum",
        enable_traces: bool = False,
    ) -> None:
        self._retry_policy = RetryPolicy(**(retry_policy or {}))
        self._metrics = MetricsCollector(
            backend=metrics_backend,
            namespace=metrics_namespace,
            enable_traces=enable_traces,
        )
        self._scheduler = Scheduler(
            max_cpu_workers=max_cpu_workers,
            max_gpu_workers=max_gpu_workers,
            max_concurrent_remote_calls=max_concurrent_remote_calls,
            tokens_per_minute_limit=tokens_per_minute_limit,
            remote_limits=remote_limits or {},
            default_timeout=default_timeout_seconds,
            retry_policy=self._retry_policy,
            metrics=self._metrics,
        )

    async def run(self, pipe: "Pipeline[..., Any]", *args: Any, **kwargs: Any) -> Any:
        run_id = str(uuid.uuid4())
        ctx = PipelineContext(scheduler=self._scheduler, run_id=run_id)
        set_current_context(ctx)
        try:
            return await pipe._execute(*args, **kwargs)
        finally:
            set_current_context(None)

    def run_sync(self, pipe: "Pipeline[..., Any]", *args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "runtime.run_sync() cannot be called inside a running event loop. "
                "Use `await runtime.run(...)` instead."
            )
        return asyncio.run(self.run(pipe, *args, **kwargs))

    async def aclose(self) -> None:
        await self._scheduler.aclose()

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
