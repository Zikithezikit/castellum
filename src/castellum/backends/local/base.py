from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LocalModelBackend(ABC):
    @abstractmethod
    def infer(self, inputs: list[Any]) -> list[Any]:
        ...

    def load(self) -> None:
        ...

    def unload(self) -> None:
        ...
