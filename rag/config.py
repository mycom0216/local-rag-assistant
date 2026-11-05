"""환경변수로 덮어쓸 수 있는 설정 모음."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma:latest")
CHAT_MODEL = os.getenv("CHAT_MODEL", "kanana2:3b")

# GPU 레이어 수. 빈 값이면 Ollama 기본값, 0이면 CPU 전용 (CUDA 호환 문제가 있을 때 사용)
NUM_GPU = os.getenv("OLLAMA_NUM_GPU", "")

DOCS_DIR =Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))      # 글자 수 기준
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

TOP_K = int(os.getenv("TOP_K", "4"))
CANDIDATES = int(os.getenv("CANDIDATES", "12"))        # 각 검색기에서 가져올 후보 수
RRF_K = int(os.getenv("RRF_K", "60"))

# 이 값보다 최고 유사도가 낮으면 LLM을 호출하지 않고 "근거 없음"으로 응답
MIN_DENSE_SCORE = float(os.getenv("MIN_DENSE_SCORE", "0.45"))
