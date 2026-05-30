from __future__ import annotations

import inspect
import warnings
from collections.abc import Generator
from typing import Any, Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from castellum.core.task import Task

R = TypeVar("R")


class MapProxy(Generic[R]):
    def __init__(
        self,
        task: Task[..., R],
        iterable: list[Any],
        batch_size: int,
        concurrency: int | None,
    ) -> None:
        self._task = task
        self._iterable = iterable
        self._batch_size = batch_size
        self._concurrency = concurrency

    def __await__(self) -> Generator[Any, None, list[R]]:
        return self._run().__await__()

    async def _run(self) -> list[R]:
        from castellum.core.context import get_current_context

        ctx = get_current_context()
        if ctx is None:
            warnings.warn(
                f"map() on {self._task} called outside a pipeline context — running sequentially with no concurrency.",
                RuntimeWarning,
                stacklevel=2,
            )
            return await self._run_direct()
        return await ctx.scheduler.map(
            self._task,
            self._iterable,
            batch_size=self._batch_size,
            concurrency=self._concurrency,
        )

    async def _run_direct(self) -> list[R]:
        results: list[R] = []
        for item in self._iterable:
            fn = self._task._fn
            r = fn(item)
            if inspect.isawaitable(r):
                r = await r
            results.append(r)
        return results
