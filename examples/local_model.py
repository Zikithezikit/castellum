"""
Example: local PyTorch model plugged in via LocalModelBackend.
"""
from __future__ import annotations

from castellum import ai_task, pipeline, Runtime
from castellum.backends.local.base import LocalModelBackend


class TorchLMBackend(LocalModelBackend):
    def __init__(self):
        self.model = None

    def load(self):
        pass

    def infer(self, prompts: list[str]) -> list[str]:
        return [f"[local] {p}" for p in prompts]


backend = TorchLMBackend()


@ai_task(kind="llm_local", device="cuda", batchable=True, max_batch_size=16, backend=backend)
def generate_local(prompts: list[str]) -> list[str]:
    return backend.infer(prompts)


@pipeline
async def local_pipeline(prompts: list[str]) -> list[str]:
    return await generate_local.map(prompts, batch_size=16)


if __name__ == "__main__":
    import asyncio
    runtime = Runtime(max_gpu_workers=1)
    results = asyncio.run(runtime.run(local_pipeline, ["hello", "world"]))
    for r in results:
        print(r)
