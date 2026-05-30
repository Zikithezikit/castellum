import pytest
from castellum import ai_task, pipeline, Runtime


@ai_task(kind="preprocess")
def chunk_docs(docs: list[str]) -> list[str]:
    return [chunk for doc in docs for chunk in doc.split("\n\n")]


@ai_task(kind="embedding", device="cpu", batchable=True, max_batch_size=8)
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return [[float(i)] for i in range(len(chunks))]


@ai_task(kind="postprocess")
async def retrieve_top_k(vectors: list[list[float]], k: int = 2) -> str:
    return "stub context"


@ai_task(kind="llm_remote", model="stub-model")
async def answer(question: str, context: str) -> str:
    return f"Answer to '{question}' given context '{context}'"


@pipeline
async def rag_pipeline(docs: list[str], question: str) -> str:
    chunks = chunk_docs(docs)
    vecs = await embed_chunks.map(chunks, batch_size=4)
    context = await retrieve_top_k(vecs, k=2)
    return await answer(question, context)


@pytest.mark.asyncio
async def test_rag_pipeline_returns_answer():
    docs = ["Hello world.\n\nThis is a test.", "Another doc.\n\nMore content."]
    runtime = Runtime(max_cpu_workers=2, max_gpu_workers=1)
    result = await runtime.run(rag_pipeline, docs, "What is this?")
    assert "Answer to" in result
    assert "stub context" in result
