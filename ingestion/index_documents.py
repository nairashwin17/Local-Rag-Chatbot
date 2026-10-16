"""CLI entry point: collect local documents, embed chunks, and persist them in Chroma."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

from config import get_settings
from ingestion.load_documents import load_local_documents
from ingestion.split_documents import split_documents
from rag.embeddings import embed_texts
from rag.retriever import get_collection


def _chunk_id(source: str, chunk_index: int, text: str) -> str:
    """Create a stable ID so rerunning ingestion does not add duplicates."""
    return hashlib.sha256(f"{source}:{chunk_index}:{text}".encode("utf-8")).hexdigest()


def _source_version(text: str) -> str:
    """Fingerprint a whole source so edited pages replace their old chunks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_changed_sources(collection, chunks):
    """Return new chunks; delete all stale chunks for a source before replacing it."""
    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.metadata["source"]].append(chunk)
    chunks_to_add = []
    for source, source_chunks in by_source.items():
        version = _source_version("".join(chunk.page_content for chunk in source_chunks))
        existing = collection.get(where={"source": source}, include=["metadatas"])
        existing_versions = {metadata.get("source_version") for metadata in existing["metadatas"]}
        if existing["ids"] and existing_versions == {version}:
            continue
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        for chunk in source_chunks:
            chunk.metadata["source_version"] = version
        chunks_to_add.extend(source_chunks)
    return chunks_to_add


def index_documents() -> tuple[int, int]:
    """Index approved local files from `docs/` and `data/raw/`.

    Changed documents replace their previous chunks; unchanged documents are skipped.
    """
    settings = get_settings()
    documents = load_local_documents(settings.raw_data_dir, settings.docs_dir)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    collection = get_collection(settings)
    new_chunks = _replace_changed_sources(collection, chunks)
    if not new_chunks:
        return len(chunks), 0
    ids = [_chunk_id(item.metadata["source"], item.metadata["chunk_index"], item.page_content) for item in new_chunks]
    collection.add(ids=ids, documents=[item.page_content for item in new_chunks], metadatas=[{key: str(value) for key, value in item.metadata.items()} for item in new_chunks], embeddings=embed_texts([item.page_content for item in new_chunks], settings))
    return len(chunks), len(new_chunks)


if __name__ == "__main__":
    logging.basicConfig(level=get_settings().log_level)
    discovered, added = index_documents()
    print(f"Processed {discovered} chunks; added {added} new chunks to ChromaDB.")
