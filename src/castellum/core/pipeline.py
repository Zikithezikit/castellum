from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class Pipeline(Generic[P, R]):
    def __init__(self, fn: Callable[P, R]) -> None:
        self._fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self._is_generator = inspect.isasyncgenfunction(fn)

    async def _execute(self, *args: Any, **kwargs: Any):
        if self._is_generator:
            return self._fn(*args, **kwargs)
        return await self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Pipeline {self.__name__!r}>"


def pipeline(fn: Callable[P, R]) -> Pipeline[P, R]:
    return Pipeline(fn)
