from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStoreBackend(ABC):
    @abstractmethod
    async def search(self, vectors: list[list[float]], *, top_k: int) -> list[Any]:
        ...

    @abstractmethod
    async def fetch_texts(self, ids: list[Any]) -> list[str]:
        ...

    async def upsert(self, ids: list[Any], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        raise NotImplementedError
