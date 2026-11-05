"""Ollama 없이 돌아가는 단위 테스트."""
from rag.chunker import chunk_documents
from rag.index import rrf, tokenize
from rag.pipeline import used_citations

DOC = """# 규정

## 휴가

연차는 15일이다.

연차는 1년 안에 써야 한다.

## 보안

비밀번호는 12자 이상이다.
"""


def test_chunks_keep_heading_path():
    chunks = chunk_documents([("a.md", DOC)], size=500, overlap=50)
    sections = {c.section for c in chunks}
    assert sections == {"규정 > 휴가", "규정 > 보안"}


def test_long_section_is_split_within_size():
    body = "\n\n".join(f"문단{i} " + "가" * 100 for i in range(10))
    chunks = chunk_documents([("a.md", "## 긴 섹션\n\n" + body)], size=300, overlap=30)
    assert len(chunks) > 1
    assert all(len(c.text) <= 330 for c in chunks)


def test_tokenize_matches_korean_particles():
    # 조사가 달라도 bigram 이 겹쳐서 키워드 검색이 된다
    assert set(tokenize("연차휴가를")) & set(tokenize("연차휴가는")) >= {"연차", "차휴", "휴가"}


def test_rrf_prefers_docs_ranked_high_in_both_lists():
    scores = rrf([[1, 2, 3], [2, 1, 4]], k=60)
    assert scores[1] > scores[3] and scores[2] > scores[4]
    assert max(scores, key=scores.get) in (1, 2)


def test_used_citations_ignores_out_of_range():
    assert used_citations("답입니다 [1] 그리고 [3] [9]", 3) == [1, 3]
