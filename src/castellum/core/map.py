from __future__ import annotations

import inspect
from typing import Any, Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from castellum.core.task import Task

R = TypeVar("R")


class MapProxy(Generic[R]):
    def __init__(
        self,
        task: "Task",
        iterable: list[Any],
        batch_size: int,
        concurrency: int | None,
    ) -> None:
        self._task = task
        self._iterable = iterable
        self._batch_size = batch_size
        self._concurrency = concurrency

    def __await__(self):
        return self._run().__await__()

    async def _run(self) -> list[R]:
        from castellum.core.context import get_current_context

        ctx = get_current_context()
        if ctx is None:
            return await self._run_direct()
        return await ctx.scheduler.map(
            self._task,
            self._iterable,
            batch_size=self._batch_size,
            concurrency=self._concurrency,
        )

    async def _run_direct(self) -> list[R]:
        results = []
        for item in self._iterable:
            r = self._task._fn(item)
            if inspect.isawaitable(r):
                r = await r
            results.append(r)
        return results
