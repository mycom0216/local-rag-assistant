"""질의 -> 검색 -> 근거 판단 -> 생성(출처 표기) 파이프라인."""
import re
from dataclasses import dataclass, field

from . import config, ollama_client
from .index import Hit, Index

NO_ANSWER = "제공된 문서에서 해당 내용의 근거를 찾을 수 없습니다."

SYSTEM_PROMPT = f"""당신은 사내 문서 기반 질의응답 어시스턴트입니다.
규칙:
1. [참고 문서]에 적힌 내용만 사용해 한국어로 1~2문장으로 답하세요.
2. 질문이 묻는 항목(예: 며칠, 몇 분, 언제까지)에 해당하는 숫자와 조건을 문서 표현 그대로 쓰세요. 질문과 다른 항목의 숫자를 쓰지 마세요.
3. 문장 끝에 근거가 된 문서 번호를 [1], [2] 형태로 표기하세요.
4. 문서에 질문과 관련된 내용이 있으면 반드시 그 내용으로 답하세요. 질문과 관련된 내용이 전혀 없을 때만 다른 말 없이 정확히 다음 문장만 출력하세요: {NO_ANSWER}
5. 문서에 없는 내용을 추측해서 덧붙이지 마세요.

예시)
[참고 문서] [1] 대형 회의실은 사용 이틀 전까지 예약하고, 승인은 총무팀이 한다. 소형 회의실은 당일 예약도 가능하다.
[질문] 대형 회의실은 언제까지 예약하나요?
답변: 대형 회의실은 사용 이틀 전까지 예약해야 합니다. [1]"""

# 최고 유사도 대비 이만큼 이상 낮은 청크는 컨텍스트에서 제외 (관련 없는 숫자가 섞여 오답을 유도함)
RELATIVE_MARGIN = 0.10
MAX_CONTEXT = 3


def select_context(hits: list[Hit]) -> list[Hit]:
    if not hits:
        return []
    top = max(h.dense for h in hits)
    kept = [h for h in hits if h.dense >= top - RELATIVE_MARGIN]
    return kept[:MAX_CONTEXT]


@dataclass
class Answer:
    answer: str
    grounded: bool
    sources: list[dict] = field(default_factory=list)
    top_dense_score: float = 0.0


def build_prompt(question: str, hits: list[Hit]) -> str:
    ctx = "\n\n".join(f"[{i}] ({h.chunk.source} > {h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(hits, 1))
    return f"[참고 문서]\n{ctx}\n\n[질문]\n{question}"


def used_citations(text: str, n: int) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n})


def ask(index: Index, question: str, mode: str = "hybrid") -> Answer:
    hits = index.search(question, k=config.TOP_K, mode=mode)
    top_dense = max((h.dense for h in hits), default=0.0)

    # 1차 방어: 검색 결과가 질문과 충분히 가깝지 않으면 LLM을 부르지 않는다.
    if not hits or top_dense < config.MIN_DENSE_SCORE:
        return Answer(NO_ANSWER, False, [], top_dense)

    hits = select_context(hits)
    text = ollama_client.chat(SYSTEM_PROMPT, build_prompt(question, hits))
    if NO_ANSWER in text:
        return Answer(NO_ANSWER, False, [], top_dense)

    cited = used_citations(text, len(hits)) or list(range(1, len(hits) + 1))
    sources = [
        {
            "ref": i,
            "source": hits[i - 1].chunk.source,
            "section": hits[i - 1].chunk.section,
            "snippet": hits[i - 1].chunk.text[:200],
            "score": round(hits[i - 1].score, 4),
        }
        for i in cited
    ]
    return Answer(text, True, sources, top_dense)
