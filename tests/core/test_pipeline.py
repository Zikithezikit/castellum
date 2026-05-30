import pytest
from castellum import ai_task, pipeline, Runtime


@pytest.mark.asyncio
async def test_simple_pipeline():
    @ai_task(kind="preprocess")
    def double(x: int) -> int:
        return x * 2

    @pipeline
    async def my_pipeline(x: int) -> int:
        return double(x)

    runtime = Runtime()
    result = await runtime.run(my_pipeline, 21)
    assert result == 42


@pytest.mark.asyncio
async def test_streaming_pipeline():
    @ai_task(kind="llm_remote", stream=True)
    async def streamer(text: str):
        for char in text:
            yield char

    @pipeline
    async def stream_pipe(text: str):
        async for chunk in streamer(text):
            yield chunk

    runtime = Runtime()
    gen = await runtime.run(stream_pipe, "abc")
    result = "".join([chunk async for chunk in gen])
    assert result == "abc"
