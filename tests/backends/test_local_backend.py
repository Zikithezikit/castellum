import os

import pytest

from castellum import ai_task, pipeline, Runtime
from castellum.backends.local.base import LocalModelBackend


class ConcatBackend(LocalModelBackend):
    def infer(self, inputs: list[str]) -> list[str]:
        return [f"processed:{x}" for x in inputs]


def test_local_model_backend_contract():
    backend = ConcatBackend()
    backend.load()
    result = backend.infer(["a", "b"])
    assert result == ["processed:a", "processed:b"]
    backend.unload()


@pytest.mark.asyncio
async def test_local_model_via_gpu_scheduler():
    backend = ConcatBackend()

    @ai_task(kind="llm_local", device="cuda", batchable=True, max_batch_size=4, backend=backend)
    def local_transform(texts: list[str]) -> list[str]:
        return backend.infer(texts)

    @pipeline
    async def pipe(items: list[str]):
        return await local_transform.map(items, batch_size=2)

    runtime = Runtime(max_gpu_workers=2)
    result = await runtime.run(pipe, ["hello", "world", "foo", "bar"])
    assert result == ["processed:hello", "processed:world", "processed:foo", "processed:bar"]
    await runtime.aclose()


_LOCAL_ENDPOINT = os.environ.get("LOCAL_INFERENCE_URL")
_LOCAL_MODEL = os.environ.get("LOCAL_INFERENCE_MODEL", "gpt-4.1-mini")
_LOCAL_API_KEY = os.environ.get("LOCAL_INFERENCE_API_KEY", "")


@pytest.mark.skipif(not _LOCAL_ENDPOINT, reason="Set LOCAL_INFERENCE_URL to test against a local inference server")
@pytest.mark.asyncio
async def test_local_inference_chat():
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=_LOCAL_ENDPOINT, api_key=_LOCAL_API_KEY)
    result = await client.chat.completions.create(
        model=_LOCAL_MODEL,
        messages=[{"role": "user", "content": "Say hello in one word"}],
    )
    text = result.choices[0].message.content or ""
    assert len(text) > 0
    await client.close()


@pytest.mark.skipif(not _LOCAL_ENDPOINT, reason="Set LOCAL_INFERENCE_URL to test against a local inference server")
@pytest.mark.asyncio
async def test_local_inference_stream():
    from openai import AsyncOpenAI
    from openai.lib.streaming.chat import ContentDeltaEvent

    client = AsyncOpenAI(base_url=_LOCAL_ENDPOINT, api_key=_LOCAL_API_KEY)
    chunks: list[str] = []
    async with client.chat.completions.stream(
        model=_LOCAL_MODEL,
        messages=[{"role": "user", "content": "Count 1 2 3"}],
    ) as stream:
        async for event in stream:
            if isinstance(event, ContentDeltaEvent) and event.delta:
                chunks.append(event.delta)
    assert len("".join(chunks)) > 0
    await client.close()
