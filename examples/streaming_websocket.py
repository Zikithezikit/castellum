"""
Example: streaming pipeline over WebSocket.

Run with:
    uvicorn streaming_websocket:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from castellum import ai_task, pipeline, Runtime
from castellum.backends.remote.openai import OpenAIClient

runtime: Runtime
llm: OpenAIClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime, llm
    llm = OpenAIClient()
    runtime = Runtime()
    yield
    await llm.aclose()
    await runtime.aclose()


app = FastAPI(lifespan=lifespan)


@ai_task(kind="llm_remote", stream=True)
async def stream_answer(prompt: str):
    async for chunk in llm.stream_chat(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    ):
        yield chunk


@pipeline
async def streaming_pipeline(prompt: str):
    async for token in stream_answer(prompt):
        yield token


@app.websocket("/stream")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    prompt = await websocket.receive_text()
    gen = await runtime.run(streaming_pipeline, prompt)
    async for token in gen:
        await websocket.send_text(token)
    await websocket.close()
