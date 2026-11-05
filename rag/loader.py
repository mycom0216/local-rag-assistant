"""문서 로더: md / txt / pdf -> (source, text)."""
from pathlib import Path


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".md", ".txt"):
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        else:
            continue
        if text.strip():
            docs.append((path.name, text))
    return docs
