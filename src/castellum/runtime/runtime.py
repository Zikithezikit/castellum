from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import AsyncGenerator
from typing import Any, TYPE_CHECKING

from castellum.core.context import PipelineContext, set_current_context
from castellum.metrics.collector import MetricsCollector, MetricsBackend
from castellum.runtime.retry import RetryPolicy
from castellum.runtime.scheduler import Scheduler

if TYPE_CHECKING:
    from castellum.core.pipeline import Pipeline


async def _with_context(
    gen: AsyncGenerator[Any, None],
    ctx: PipelineContext,
) -> AsyncGenerator[Any, None]:
    try:
        set_current_context(ctx)
        async for item in gen:
            yield item
    finally:
        set_current_context(None)


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
        metrics_backend: str | MetricsBackend = MetricsBackend.NONE,
        metrics_namespace: str = "castellum",
        enable_traces: bool = False,
    ) -> None:
        if retry_policy:
            self._retry_policy = RetryPolicy(
                max_retries=retry_policy.get("max_retries", 3),
                base_delay=retry_policy.get("base_delay", 0.5),
                max_delay=retry_policy.get("max_delay", 30.0),
                jitter=retry_policy.get("jitter", True),
                retryable_exceptions=retry_policy.get("retryable_exceptions"),
            )
        else:
            self._retry_policy = RetryPolicy()
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
            result = await pipe._execute(*args, **kwargs)
            if pipe._is_generator:
                return _with_context(result, ctx)
            return result
        finally:
            if not pipe._is_generator:
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

        async def _collect() -> Any:
            result = await self.run(pipe, *args, **kwargs)
            if inspect.isasyncgen(result):
                return [token async for token in result]
            return result

        return asyncio.run(_collect())

    async def aclose(self) -> None:
        await self._scheduler.aclose()

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return (
            f"<Runtime cpu={self._scheduler._executor._max_workers}"
            f" retry={self._retry_policy.max_retries}>"
        )
