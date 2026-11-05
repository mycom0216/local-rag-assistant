"""헤딩 인식 청킹.

마크다운 헤딩(#, ##)을 경계로 섹션을 나누고, 섹션이 길면 문단 단위로 묶어
chunk_size 이하로 자른다. 모든 청크 앞에 "문서 > 섹션" 경로를 붙여 검색 품질을 높인다.
"""
import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    id: int
    source: str
    section: str
    text: str

    @property
    def display(self) -> str:
        return f"[{self.source} > {self.section}]\n{self.text}"


def _sections(text: str) -> list[tuple[str, str]]:
    sections, path, buf = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(path) or "본문", body))
        buf.clear()

    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return sections


def _split_body(body: str, size: int, overlap: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces, cur = [], ""
    for p in paras:
        if len(p) > size:  # 긴 문단은 글자 수로 강제 분할
            if cur:
                pieces.append(cur)
                cur = ""
            step = max(size - overlap, 1)
            pieces.extend(p[i : i + size] for i in range(0, len(p), step))
            continue
        if cur and len(cur) + len(p) + 2 > size:
            pieces.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n\n" + p).strip()
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
    if cur:
        pieces.append(cur)
    return pieces


def chunk_documents(docs: list[tuple[str, str]], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in docs:
        for section, body in _sections(text):
            for piece in _split_body(body, size, overlap):
                chunks.append(Chunk(len(chunks), source, section, piece))
    return chunks
