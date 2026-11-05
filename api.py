"""FastAPI 서버.

  uvicorn api:app --reload
  POST /ask      {"question": "...", "mode": "hybrid|dense|bm25"}
  POST /reindex  문서를 다시 읽어 인덱스 재구축
  GET  /health
"""
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import config, ollama_client, pipeline
from rag.index import Index

state: dict = {"index": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state["index"] = Index.load()
    except FileNotFoundError:
        state["index"] = None  # /reindex 로 생성
    yield


app = FastAPI(title="Local RAG Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"


@app.get("/health")
def health():
    idx = state["index"]
    return {
        "indexed_chunks": len(idx.chunks) if idx else 0,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.post("/reindex")
def reindex():
    try:
        state["index"] = Index.build()
    except Exception as e:  # Ollama 미기동 등
        raise HTTPException(503, f"인덱싱 실패: {e}")
    return {"indexed_chunks": len(state["index"].chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    if state["index"] is None:
        raise HTTPException(409, "인덱스가 없습니다. 먼저 POST /reindex 를 호출하세요.")
    try:
        return asdict(pipeline.ask(state["index"], req.question, req.mode))
    except ollama_client.httpx.HTTPError as e:
        raise HTTPException(503, f"Ollama 호출 실패: {e}")
