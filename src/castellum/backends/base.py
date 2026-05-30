from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        ...

    @abstractmethod
    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        ...

    async def aclose(self) -> None:
        ...
