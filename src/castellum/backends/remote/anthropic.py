from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from castellum.backends.base import LLMClient


class AnthropicClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        timeout: float = 60.0,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("Install 'castellum[anthropic]' to use AnthropicClient.") from e

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self._default_model = default_model

    async def chat(
        self,
        *,
        model: str | None = None,
        messages: list[Any],
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        response = await self._client.messages.create(
            model=model or self._default_model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs,
        )
        return str(response.content[0].text)

    async def stream_chat(
        self,
        *,
        model: str | None = None,
        messages: list[Any],
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        async with self._client.messages.stream(
            model=model or self._default_model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def aclose(self) -> None:
        await self._client.close()
