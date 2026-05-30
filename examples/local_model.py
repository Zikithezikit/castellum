"""
Example: local model plugged in via LocalModelBackend.

Replace the stub infer() with your actual model inference logic.
"""

from __future__ import annotations

from castellum import ai_task, pipeline, Runtime
from castellum.backends.local.base import LocalModelBackend


class LocalEmbedder(LocalModelBackend):
    def __init__(self):
        self.model = None

    def load(self):
        pass

    def infer(self, inputs: list[str]) -> list[list[float]]:
        return [[0.0]] * len(inputs)


backend = LocalEmbedder()


@ai_task(kind="embedding", device="cuda", batchable=True, max_batch_size=16, backend=backend)
def embed(texts: list[str]) -> list[list[float]]:
    return backend.infer(texts)


@pipeline
async def embedding_pipeline(texts: list[str]) -> list[list[float]]:
    return await embed.map(texts, batch_size=16)


if __name__ == "__main__":
    import asyncio

    runtime = Runtime(max_gpu_workers=1)
    results = asyncio.run(runtime.run(embedding_pipeline, ["hello", "world"]))
    for r in results:
        print(r)
