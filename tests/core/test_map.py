import pytest
from castellum import ai_task, pipeline, Runtime


@pytest.mark.asyncio
async def test_map_non_batchable():
    @ai_task(kind="preprocess")
    def square(x: int) -> int:
        return x ** 2

    @pipeline
    async def pipe(items):
        return await square.map(items)

    runtime = Runtime()
    result = await runtime.run(pipe, [1, 2, 3, 4])
    assert result == [1, 4, 9, 16]


@pytest.mark.asyncio
async def test_map_batchable():
    calls = []

    @ai_task(kind="embedding", batchable=True, max_batch_size=3)
    def embed(batch: list[str]) -> list[list[float]]:
        calls.append(len(batch))
        return [[0.1, 0.2]] * len(batch)

    @pipeline
    async def pipe(items):
        return await embed.map(items, batch_size=3)

    runtime = Runtime()
    result = await runtime.run(pipe, ["a", "b", "c", "d", "e"])
    assert len(result) == 5
    assert sorted(calls) == [2, 3]
