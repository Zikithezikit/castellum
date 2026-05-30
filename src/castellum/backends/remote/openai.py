from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from castellum.backends.base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = "gpt-4.1-mini",
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Install 'castellum[openai]' to use OpenAIClient.") from e

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._default_model = default_model

    async def chat(
        self,
        *,
        model: str | None = None,
        messages: list[Any],
        **kwargs: Any,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=model or self._default_model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def stream_chat(
        self,
        *,
        model: str | None = None,
        messages: list[Any],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        from openai.lib.streaming.chat import ContentDeltaEvent

        async with self._client.chat.completions.stream(
            model=model or self._default_model,
            messages=messages,
            **kwargs,
        ) as stream:
            async for event in stream:
                if isinstance(event, ContentDeltaEvent) and event.delta:
                    yield event.delta

    async def aclose(self) -> None:
        await self._client.close()
