from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from castellum.backends.base import LLMClient


class AnthropicClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        timeout: float = 60.0,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("Install 'castellum[anthropic]' to use AnthropicClient.") from e

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
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
        for block in response.content:
            if hasattr(block, "text") and block.text:
                return block.text  # type: ignore[no-any-return]
        return ""

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
