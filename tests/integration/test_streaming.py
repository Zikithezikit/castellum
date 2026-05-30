import pytest
from castellum import ai_task, pipeline, Runtime


@ai_task(kind="llm_remote", stream=True)
async def stream_tokens(prompt: str):
    for char in f"streaming: {prompt}":
        yield char


@pipeline
async def streaming_pipeline(prompt: str):
    async for token in stream_tokens(prompt):
        yield token


@pytest.mark.asyncio
async def test_streaming_pipeline_yields_tokens():
    runtime = Runtime()
    gen = await runtime.run(streaming_pipeline, "hello")
    result = "".join([chunk async for chunk in gen])
    assert result == "streaming: hello"
