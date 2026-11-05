"""하이브리드 검색 인덱스: BM25(키워드) + 벡터(의미) -> RRF 결합."""
import json
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from . import config, ollama_client
from .chunker import Chunk, chunk_documents
from .loader import load_documents

_WORD = re.compile(r"[가-힣a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """형태소 분석기 없이 한국어를 다루기 위한 단어 + 문자 bigram 토크나이저.

    '연차휴가를' 과 '연차휴가는' 처럼 조사만 다른 단어도 bigram 이 겹쳐 매칭된다.
    """
    tokens = []
    for w in _WORD.findall(text.lower()):
        tokens.append(w)
        if len(w) >= 2:
            tokens.extend(w[i : i + 2] for i in range(len(w) - 1))
    return tokens


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion. 서로 다른 점수 스케일을 순위만으로 결합한다."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float  # 질의-청크 코사인 유사도 (근거 부족 판단에 사용)


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = BM25Okapi([tokenize(c.section + " " + c.text) for c in chunks])

    # ---- 구축 / 저장 / 로드 -------------------------------------------------
    @classmethod
    def build(cls) -> "Index":
        docs = load_documents(config.DOCS_DIR)
        chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            raise RuntimeError(f"문서가 없습니다: {config.DOCS_DIR}")
        vectors = ollama_client.embed([f"{c.section}\n{c.text}" for c in chunks])
        idx = cls(chunks, vectors)
        idx.save()
        return idx

    def save(self) -> None:
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        meta = [c.__dict__ for c in self.chunks]
        (config.INDEX_DIR / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        np.save(config.INDEX_DIR / "vectors.npy", self.vectors)

    @classmethod
    def load(cls) -> "Index":
        meta = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        return cls([Chunk(**m) for m in meta], np.load(config.INDEX_DIR / "vectors.npy"))

    # ---- 검색 ------------------------------------------------------------
    def search(self, query: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[Hit]:
        qvec = ollama_client.embed([query], is_query=True)[0]
        dense_scores = self.vectors @ qvec
        dense_rank = list(np.argsort(-dense_scores)[: config.CANDIDATES])

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = [int(i) for i in np.argsort(-bm25_scores)[: config.CANDIDATES] if bm25_scores[i] > 0]

        if mode == "dense":
            fused = {int(i): float(dense_scores[i]) for i in dense_rank}
        elif mode == "bm25":
            fused = {i: float(bm25_scores[i]) for i in bm25_rank}
        else:
            fused = rrf([[int(i) for i in dense_rank], bm25_rank], config.RRF_K)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], float(dense_scores[i])) for i in top]
