#!/usr/bin/env python3
"""
RAG pipeline demo showcasing castellum's core features:
  - @ai_task with kind/device/batchable metadata → automatic dispatch
  - task.map() → batched fan-out through scheduler
  - GPU vs CPU vs remote dispatch (semaphores, thread pool)
  - Rate limiting and retry policy
  - Streaming tasks (bypass scheduler, call fn directly)
  - OpenAIClient via OPENAI_BASE_URL / OPENAI_API_KEY env vars
"""

import os
import asyncio
from collections.abc import AsyncGenerator
from castellum import ai_task, pipeline, Runtime, TaskKind

_LLM_URL = os.environ.get("OPENAI_BASE_URL")
_LLM_KEY = os.environ.get("OPENAI_API_KEY")

if _LLM_URL and _LLM_KEY:
    from castellum.backends.remote.openai import OpenAIClient
    _client = OpenAIClient(api_key=_LLM_KEY)

    @ai_task(kind=TaskKind.LLM_REMOTE, model="gemini-3")
    async def answer(question: str, context: str) -> str:
        return await _client.chat(
            model="gemini-3",
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQ: {question}\nA:"}],
        )

    @ai_task(kind=TaskKind.LLM_REMOTE, model="gemini-3", stream=True)
    async def stream_answer(question: str, context: str) -> AsyncGenerator[str, None]:
        async for chunk in _client.stream_chat(
            model="gemini-3",
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQ: {question}\nA:"}],
        ):
            yield chunk
else:
    @ai_task(kind=TaskKind.LLM_REMOTE, model="mock")
    async def answer(question: str, context: str) -> str:
        return f"[mock] Context ({len(context)} chars) → answer to: {question}"

    @ai_task(kind=TaskKind.LLM_REMOTE, model="mock", stream=True)
    async def stream_answer(question: str, context: str) -> AsyncGenerator[str, None]:
        words = f"[mock streaming] Based on: {context[:30]}... answering: {question}".split()
        for w in words:
            yield w + " "

# --- CPU-bound preprocessing: dispatched to thread pool ---

@ai_task(kind=TaskKind.PREPROCESS, device="cpu")
def chunk_text(text: str, chunk_size: int = 60) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# --- GPU-bound embedding: guarded by GPU semaphore → max 1 concurrent ---

@ai_task(kind=TaskKind.EMBEDDING, device="cuda", batchable=True, max_batch_size=4)
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return [[float(hash(c) % 10_000) / 10_000] for c in chunks]

# --- Postprocessing (plain async, no blocking) ---

@ai_task(kind=TaskKind.POSTPROCESS)
async def pick_top(vectors: list[list[float]], k: int = 2) -> list[int]:
    scored = sorted(enumerate(vectors), key=lambda x: x[1][0], reverse=True)
    return [i for i, _ in scored[:k]]

@ai_task(kind=TaskKind.POSTPROCESS)
async def join_chunks(chunks: list[str], indices: list[int]) -> str:
    return "\n".join(chunks[i] for i in indices)

# --- Pipelines ---

@pipeline
async def rag_pipeline(docs: list[str], question: str) -> str:
    chunks = chunk_text(" ".join(docs))
    vecs = await embed_chunks.map(chunks, preferred_batch_size=4)
    top = await pick_top(vecs, k=2)  # type: ignore[arg-type]
    context = await join_chunks(chunks, top)
    return await answer(question, context)


@pipeline
async def streaming_pipeline(docs: list[str], question: str) -> AsyncGenerator[str, None]:
    chunks = chunk_text(" ".join(docs))
    vecs = await embed_chunks.map(chunks, preferred_batch_size=4)
    top = await pick_top(vecs, k=2)  # type: ignore[arg-type]
    context = await join_chunks(chunks, top)
    async for token in stream_answer(question, context):
        yield token


async def main() -> None:
    runtime = Runtime(
        max_cpu_workers=4,
        max_gpu_workers=1,
        max_concurrent_remote_calls=5,
        tokens_per_minute_limit=200_000,
        retry_policy={"max_retries": 2, "base_delay": 0.3},
    )

    docs = [
        "Castellum is an AI pipeline orchestration library for Python.",
        "It dispatches CPU tasks to a thread pool, GPU tasks behind a semaphore, "
        "and remote LLM calls through rate limiters with retry logic.",
        "Tasks are decorated with @ai_task(kind, device, batchable) metadata.",
        "Pipelines are async functions decorated with @pipeline.",
        "The Runtime wires the event loop, thread pool, and rate limiters together.",
    ]
    question = "What does castellum do?"

    print("=" * 60)
    print("1. Standard RAG pipeline (batch map + LLM call)")
    print("=" * 60)
    result = await runtime.run(rag_pipeline, docs, question)
    print(result)

    print()
    print("=" * 60)
    print("2. Streaming pipeline (async generator)")
    print("=" * 60)
    gen = await runtime.run(streaming_pipeline, docs, question)
    async for token in gen:
        print(token, end="", flush=True)
    print()

    await runtime.aclose()


if __name__ == "__main__":
    asyncio.run(main())
