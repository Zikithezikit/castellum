from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TYPE_CHECKING

from castellum.core.task import Task, TaskKind
from castellum.runtime.rate_limiter import ModelRateLimiter, RateLimiter
from castellum.runtime.retry import RetryPolicy

if TYPE_CHECKING:
    from castellum.metrics.collector import MetricsCollector


class Scheduler:
    def __init__(
        self,
        *,
        max_cpu_workers: int,
        max_gpu_workers: int,
        max_concurrent_remote_calls: int,
        tokens_per_minute_limit: int | None,
        remote_limits: dict[str, dict[str, int]],
        default_timeout: float,
        retry_policy: RetryPolicy,
        metrics: "MetricsCollector",
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_cpu_workers, thread_name_prefix="castellum-cpu")
        self._gpu_sem = asyncio.Semaphore(max_gpu_workers)
        self._remote_sem = asyncio.Semaphore(max_concurrent_remote_calls)
        self._global_rate_limiter = RateLimiter(tokens_per_minute=tokens_per_minute_limit) if tokens_per_minute_limit else None
        self._model_limiters: dict[str, ModelRateLimiter] = {
            model: ModelRateLimiter(**limits)
            for model, limits in remote_limits.items()
        }
        self._default_timeout = default_timeout
        self._retry_policy = retry_policy
        self._metrics = metrics

    async def submit(self, task: Task[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        timeout = task.meta.timeout or self._default_timeout
        retry_policy = RetryPolicy(**(task.meta.retry_policy or {})) if task.meta.retry_policy else self._retry_policy

        async def _dispatch() -> Any:
            kind = task.meta.kind
            if kind == TaskKind.LLM_REMOTE:
                return await self._run_remote(task, args, kwargs)
            elif kind in (TaskKind.EMBEDDING, TaskKind.LLM_LOCAL) and task.meta.device != "cpu":
                return await self._run_gpu(task, args, kwargs)
            elif inspect.iscoroutinefunction(task._fn) or inspect.isasyncgenfunction(task._fn):
                return await self._run_async(task, args, kwargs)
            else:
                return await self._run_cpu(task, args, kwargs)

        with self._metrics.task_timer(task.__name__):
            return await asyncio.wait_for(
                retry_policy.execute(_dispatch),
                timeout=timeout,
            )

    async def map(
        self,
        task: Task[..., Any],
        items: list[Any],
        *,
        batch_size: int,
        concurrency: int | None,
    ) -> list[Any]:
        if task.meta.batchable:
            return await self._map_batched(task, items, batch_size=batch_size, concurrency=concurrency)
        else:
            return await self._map_individual(task, items, concurrency=concurrency)

    async def _map_batched(self, task: Task[..., Any], items: list[Any], *, batch_size: int, concurrency: int | None) -> list[Any]:
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        sem = asyncio.Semaphore(concurrency) if concurrency else None

        async def run_batch(batch: list[Any]) -> Any:
            if sem:
                async with sem:
                    return await self.submit(task, (batch,), {})
            return await self.submit(task, (batch,), {})

        async with asyncio.TaskGroup() as tg:
            coros = [tg.create_task(run_batch(b)) for b in batches]

        result: list[Any] = []
        for t in coros:
            r = t.result()
            if isinstance(r, list):
                result.extend(r)
            else:
                result.append(r)
        return result

    async def _map_individual(self, task: Task[..., Any], items: list[Any], *, concurrency: int | None) -> list[Any]:
        sem = asyncio.Semaphore(concurrency) if concurrency else None

        async def run_one(item: Any) -> Any:
            if sem:
                async with sem:
                    return await self.submit(task, (item,), {})
            return await self.submit(task, (item,), {})

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run_one(item)) for item in items]

        return [t.result() for t in tasks]

    async def _run_cpu(self, task: Task[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: task._fn(*args, **kwargs),
        )

    async def _run_gpu(self, task: Task[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        async with self._gpu_sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: task._fn(*args, **kwargs),
            )

    async def _run_async(self, task: Task[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return await task._fn(*args, **kwargs)

    async def _run_remote(self, task: Task[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        model = task.meta.model

        async with self._remote_sem:
            if model and model in self._model_limiters:
                await self._model_limiters[model].acquire()
            if self._global_rate_limiter:
                await self._global_rate_limiter.acquire()
            return await task._fn(*args, **kwargs)

    async def aclose(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
