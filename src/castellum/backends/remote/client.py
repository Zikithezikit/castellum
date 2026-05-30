from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx


class AsyncHTTPClient:
    def __init__(self, *, base_url: str, api_key: str, timeout: float = 60.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def post(self, path: str, *, json: dict[str, Any]) -> Any:
        response = await self._client.post(path, json=json)
        response.raise_for_status()
        return response.json()

    @asynccontextmanager
    async def stream(self, path: str, *, json: dict[str, Any]) -> AsyncIterator[Any]:
        async with self._client.stream("POST", path, json=json) as response:
            yield response

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncHTTPClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
