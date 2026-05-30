"""
Example: RAG pipeline served via FastAPI.

Run with:
    uvicorn rag_fastapi:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from castellum import ai_task, pipeline, Runtime

runtime: Runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    runtime = Runtime(
        max_cpu_workers=8,
        max_gpu_workers=2,
        max_concurrent_remote_calls=10,
        tokens_per_minute_limit=80_000,
        remote_limits={
            "gpt-4.1-mini": {"requests_per_minute": 1500, "tokens_per_minute": 100_000}
        },
        retry_policy={"max_retries": 3, "base_delay": 0.5},
        metrics_backend="prometheus",
    )
    yield
    await runtime.aclose()


app = FastAPI(lifespan=lifespan)


@ai_task(kind="preprocess")
def chunk_docs(docs: list[str]) -> list[str]:
    return [chunk for doc in docs for chunk in doc.split("\n\n")]


@ai_task(kind="embedding", device="cuda", batchable=True, max_batch_size=256)
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return [[0.0]] * len(chunks)


@ai_task(kind="postprocess")
async def retrieve_top_k(vectors: list[list[float]], k: int = 5) -> str:
    return "stub context"


@ai_task(kind="llm_remote", model="gpt-4.1-mini")
async def answer(question: str, context: str) -> str:
    return f"[stub answer] Q: {question}"


@pipeline
async def rag_pipeline(docs: list[str], question: str) -> str:
    chunks = chunk_docs(docs)
    vecs = await embed_chunks.map(chunks, preferred_batch_size=128)
    context = await retrieve_top_k(vecs, k=5)
    return await answer(question, context)


class RAGRequest(BaseModel):
    docs: list[str]
    question: str


@app.post("/rag")
async def rag_endpoint(req: RAGRequest) -> dict:
    result = await runtime.run(rag_pipeline, req.docs, req.question)
    return {"answer": result}
