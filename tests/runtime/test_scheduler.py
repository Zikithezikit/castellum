import time

import pytest
from castellum import ai_task, pipeline, Runtime


@pytest.mark.asyncio
async def test_concurrent_cpu_tasks():
    @ai_task(kind="preprocess")
    def work(x: int) -> int:
        return x + 1

    @pipeline
    async def pipe(items):
        return await work.map(items)

    runtime = Runtime(max_cpu_workers=4)
    result = await runtime.run(pipe, list(range(10)))
    assert sorted(result) == list(range(1, 11))


@pytest.mark.asyncio
async def test_gpu_semaphore_limits_concurrency():
    concurrent = []
    peak = [0]

    @ai_task(kind="embedding", device="cuda")
    def gpu_work(x: int) -> int:
        concurrent.append(1)
        peak[0] = max(peak[0], len(concurrent))
        time.sleep(0.02)
        concurrent.pop()
        return x

    @pipeline
    async def pipe(items):
        return await gpu_work.map(items)

    runtime = Runtime(max_gpu_workers=2)
    await runtime.run(pipe, list(range(6)))
    assert peak[0] <= 2
