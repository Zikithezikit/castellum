from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


class BatchCollector:
    def __init__(
        self,
        handler: Callable[[list[Any]], Any],
        *,
        max_batch_size: int = 128,
        flush_timeout: float = 0.005,
    ) -> None:
        self._handler = handler
        self._max = max_batch_size
        self._timeout = flush_timeout
        self._pending: list[tuple[Any, asyncio.Future[Any]]] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

    async def add(self, item: Any) -> Any:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()

        async with self._lock:
            self._pending.append((item, fut))
            if len(self._pending) >= self._max:
                await self._flush_locked()
            elif self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._schedule_flush())

        return await fut

    async def _schedule_flush(self) -> None:
        await asyncio.sleep(self._timeout)
        async with self._lock:
            if self._pending:
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        batch = self._pending[:]
        self._pending.clear()

        items = [item for item, _ in batch]
        futs = [fut for _, fut in batch]

        try:
            results = await self._handler(items)
            for fut, result in zip(futs, results):
                if not fut.done():
                    fut.set_result(result)
        except Exception as exc:
            for fut in futs:
                if not fut.done():
                    fut.set_exception(exc)
