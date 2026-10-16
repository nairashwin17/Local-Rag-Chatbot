"""Load approved local documents into small, source-aware records."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}
# Project architecture notes explain the application; they are not approved knowledge sources.
NON_SOURCE_FILENAMES = {"architecture.md"}


def _read_file(path: Path) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".docx":
        import docx2txt

        return docx2txt.process(str(path)) or ""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_local_documents(*directories: Path) -> list[Document]:
    """Read approved local files and preserve metadata needed for citations."""
    documents: list[Document] = []
    seen_paths: set[Path] = set()
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in SUPPORTED_SUFFIXES
                or path.name in NON_SOURCE_FILENAMES
            ):
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            text = _read_file(path).strip()
            if not text:
                continue
            documents.append(Document(page_content=text, metadata={
                "title": path.stem.replace("_", " "),
                "source": str(resolved),
                "source_type": path.suffix.lower().lstrip("."),
                "knowledge_domain": "local",
                "last_updated": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            }))
    return documents
