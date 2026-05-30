"""
Example: streaming pipeline over WebSocket.

Run with:
    uvicorn streaming_websocket:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from castellum import ai_task, pipeline, Runtime

runtime: Runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    runtime = Runtime()
    yield
    await runtime.aclose()


app = FastAPI(lifespan=lifespan)


@ai_task(kind="llm_remote", stream=True)
async def stream_tokens(prompt: str):
    for char in f"Response to: {prompt}":
        import asyncio
        await asyncio.sleep(0.01)
        yield char


@pipeline
async def streaming_pipeline(prompt: str):
    async for token in stream_tokens(prompt):
        yield token


@app.websocket("/stream")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    prompt = await websocket.receive_text()
    gen = await runtime.run(streaming_pipeline, prompt)
    async for token in gen:
        await websocket.send_text(token)
    await websocket.close()
