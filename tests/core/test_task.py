import pytest
from castellum import ai_task, Task, TaskKind


def test_bare_decorator():
    @ai_task
    def fn(x: int) -> int:
        return x * 2

    assert isinstance(fn, Task)
    assert fn.meta.kind == TaskKind.PREPROCESS


def test_parametrised_decorator():
    @ai_task(kind="embedding", device="cuda", batchable=True, max_batch_size=64)
    def embed(chunks):
        return [[0.1]] * len(chunks)

    assert embed.meta.kind == TaskKind.EMBEDDING
    assert embed.meta.device == "cuda"
    assert embed.meta.batchable is True
    assert embed.meta.max_batch_size == 64


def test_direct_call_outside_pipeline():
    @ai_task(kind="preprocess")
    def double(x: int) -> int:
        return x * 2

    assert double(5) == 10


@pytest.mark.asyncio
async def test_async_task_direct_call():
    @ai_task(kind="llm_remote", model="test-model")
    async def echo(text: str) -> str:
        return text

    result = await echo("hello")
    assert result == "hello"
