"""Split documents into retrieval-sized chunks while retaining source metadata."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Create deterministic chunks and attach a chunk number to each source."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for document in documents:
        for index, chunk in enumerate(splitter.split_documents([document])):
            chunk.metadata = {**document.metadata, "chunk_index": index}
            chunks.append(chunk)
    return chunks
