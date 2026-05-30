# castellum

> AI pipeline orchestration with automatic async I/O, CPU, and GPU management.

## Install

```bash
pip install castellum
# With OpenAI support:
pip install "castellum[openai]"
# With Anthropic support:
pip install "castellum[anthropic]"
# Everything:
pip install "castellum[all]"
```

## Quick start

```python
from castellum import ai_task, pipeline, Runtime

@ai_task(kind="preprocess")
def chunk_docs(docs: list[str]) -> list[str]:
    return [c for doc in docs for c in doc.split("\n\n")]

@ai_task(kind="embedding", device="cuda", batchable=True, max_batch_size=128)
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return local_embedder.encode(chunks)   # your model here

@ai_task(kind="llm_remote", model="gpt-4.1-mini")
async def answer(question: str, context: str) -> str:
    return await openai_client.chat(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": f"{context}\n\n{question}"}],
    )

@pipeline
async def rag_pipeline(docs: list[str], question: str) -> str:
    chunks = chunk_docs(docs)
    vecs = await embed_chunks.map(chunks, preferred_batch_size=64)
    context = "\n".join(str(v) for v in vecs[:5])
    return await answer(question, context)

runtime = Runtime(
    max_cpu_workers=8,
    max_gpu_workers=2,
    max_concurrent_remote_calls=10,
    tokens_per_minute_limit=80_000,
)

# Inside async code:
result = await runtime.run(rag_pipeline, docs, question)

# Outside async code:
result = runtime.run_sync(rag_pipeline, docs, question)
```

## Core concepts

| Concept | Description |
|---------|-------------|
| `@ai_task` | Decorates a function with scheduling metadata (`kind`, `device`, `batchable`, etc.) |
| `@pipeline` | Decorates an async function or async generator as an orchestrated flow |
| `task.map(items)` | Fan-out a task over a collection with automatic batching |
| `Runtime` | Wires the event loop, thread pool, GPU workers, and rate limiters |

## License

MIT
