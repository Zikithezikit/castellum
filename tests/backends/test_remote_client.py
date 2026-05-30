import os

import pytest

pytestmark = pytest.mark.asyncio

_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


@pytest.mark.skipif(
    not _OPENAI_BASE_URL or not _OPENAI_API_KEY,
    reason="Set OPENAI_BASE_URL and OPENAI_API_KEY env vars to test remote clients",
)
async def test_openai_chat():
    from castellum.backends.remote.openai import OpenAIClient

    client = OpenAIClient(api_key=_OPENAI_API_KEY)
    result = await client.chat(
        model="gemini-3",
        messages=[{"role": "user", "content": "Say hello in one word"}],
    )
    assert isinstance(result, str)
    assert len(result) > 0
    await client.aclose()


@pytest.mark.skipif(
    not _OPENAI_BASE_URL or not _OPENAI_API_KEY,
    reason="Set OPENAI_BASE_URL and OPENAI_API_KEY env vars to test remote clients",
)
async def test_openai_stream():
    from castellum.backends.remote.openai import OpenAIClient

    client = OpenAIClient(api_key=_OPENAI_API_KEY)
    chunks = []
    async for chunk in client.stream_chat(
        model="gemini-3",
        messages=[{"role": "user", "content": "Count 1 2 3"}],
    ):
        chunks.append(chunk)
    assert len("".join(chunks)) > 0
    await client.aclose()
