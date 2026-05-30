from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class Pipeline(Generic[P, R]):
    def __init__(self, fn: Callable[P, R]) -> None:
        self._fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self._is_generator = inspect.isasyncgenfunction(fn)
        self._is_async = inspect.iscoroutinefunction(fn) or self._is_generator

    async def _execute(self, *args: Any, **kwargs: Any) -> Any:
        if self._is_generator:
            return self._fn(*args, **kwargs)
        result = self._fn(*args, **kwargs)
        if self._is_async:
            return await cast(Awaitable[Any], result)
        return result

    def __repr__(self) -> str:
        return f"<Pipeline {self.__name__!r} async={self._is_async}>"


def pipeline(fn: Callable[P, R]) -> Pipeline[P, R]:
    return Pipeline(fn)
