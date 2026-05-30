from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, ParamSpec, TypeVar, TYPE_CHECKING, overload

if TYPE_CHECKING:
    from castellum.core.map import MapProxy

P = ParamSpec("P")
R = TypeVar("R")


class TaskKind(StrEnum):
    PREPROCESS = "preprocess"
    EMBEDDING = "embedding"
    LLM_REMOTE = "llm_remote"
    LLM_LOCAL = "llm_local"
    POSTPROCESS = "postprocess"


@dataclass(frozen=True)
class TaskMeta:
    kind: TaskKind
    device: str = "cpu"
    batchable: bool = False
    max_batch_size: int = 1
    model: str | None = None
    stream: bool = False
    backend: Any = None
    timeout: float | None = None
    retry_policy: dict[str, Any] | None = None


class Task(Generic[P, R]):
    def __init__(self, fn: Callable[P, R], meta: TaskMeta) -> None:
        self._fn = fn
        self.meta = meta
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        from castellum.core.context import get_current_context

        ctx = get_current_context()
        if ctx is None:
            return self._fn(*args, **kwargs)
        if inspect.iscoroutinefunction(self._fn) or inspect.isasyncgenfunction(self._fn):
            return ctx.scheduler.submit(self, args, kwargs)  # type: ignore[return-value]
        return self._fn(*args, **kwargs)

    def map(
        self,
        iterable: Iterable[Any],
        *,
        batch_size: int | None = None,
        preferred_batch_size: int | None = None,
        concurrency: int | None = None,
    ) -> MapProxy[R]:
        from castellum.core.map import MapProxy

        effective_batch = batch_size or preferred_batch_size or self.meta.max_batch_size
        return MapProxy(
            task=self,
            iterable=list(iterable),
            batch_size=effective_batch,
            concurrency=concurrency,
        )

    def __repr__(self) -> str:
        return f"<Task {self.__name__!r} kind={self.meta.kind}>"


@overload
def ai_task(fn: Callable[P, R]) -> Task[P, R]: ...


@overload
def ai_task(
    *,
    kind: str | TaskKind = TaskKind.PREPROCESS,
    device: str = "cpu",
    batchable: bool = False,
    max_batch_size: int = 1,
    model: str | None = None,
    stream: bool = False,
    backend: Any = None,
    timeout: float | None = None,
    retry_policy: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Task[P, R]]: ...


def ai_task(
    fn: Callable[P, R] | None = None,
    *,
    kind: str | TaskKind = TaskKind.PREPROCESS,
    device: str = "cpu",
    batchable: bool = False,
    max_batch_size: int = 1,
    model: str | None = None,
    stream: bool = False,
    backend: Any = None,
    timeout: float | None = None,
    retry_policy: dict[str, Any] | None = None,
) -> Task[P, R] | Callable[[Callable[P, R]], Task[P, R]]:
    meta = TaskMeta(
        kind=TaskKind(kind),
        device=device,
        batchable=batchable,
        max_batch_size=max_batch_size,
        model=model,
        stream=stream,
        backend=backend,
        timeout=timeout,
        retry_policy=retry_policy,
    )

    def decorator(func: Callable[P, R]) -> Task[P, R]:
        return Task(func, meta)

    if fn is not None:
        return decorator(fn)
    return decorator
