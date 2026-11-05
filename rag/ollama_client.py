"""Ollama REST 클라이언트 (임베딩 + 생성)."""
import httpx
import numpy as np

from . import config

# embeddinggemma 권장 프롬프트 형식 (검색용 비대칭 임베딩)
DOC_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def _gpu_options() -> dict:
    return {"num_gpu": int(config.NUM_GPU)} if config.NUM_GPU != "" else {}


def embed(texts: list[str], is_query: bool = False, batch: int = 16) -> np.ndarray:
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    out = []
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(texts), batch):
            r = client.post(
                f"{config.OLLAMA_URL}/api/embed",
                json={
                    "model": config.EMBED_MODEL,
                    "input": [prefix + t for t in texts[i : i + batch]],
                    "options": _gpu_options(),
                },
            )
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
    return _normalize(np.array(out, dtype=np.float32))


def chat(system: str, user: str, temperature: float = 0.1) -> str:
    r = httpx.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.CHAT_MODEL,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096, **_gpu_options()},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
