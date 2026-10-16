"""Hybrid Chroma and BM25 retrieval, kept independent from answer generation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import chromadb
from rank_bm25 import BM25Okapi

from config import Settings
from rag.embeddings import embed_texts

COLLECTION_NAME = "curious_learning_documents"


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and the metadata shown in citations."""

    text: str
    title: str
    source: str
    source_type: str
    score: float
    chunk_index: int = 0


def _where_clause(source_types: Iterable[str] | None, domains: Iterable[str] | None) -> dict | None:
    clauses = []
    if source_types:
        clauses.append({"source_type": {"$in": list(source_types)}})
    if domains:
        clauses.append({"knowledge_domain": {"$in": list(domains)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def get_collection(settings: Settings):
    """Open the project's persistent Chroma collection."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token]


def search_documents(
    query: str,
    settings: Settings,
    k: int | None = None,
    source_types: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
) -> list[SearchResult]:
    """Combine semantic and exact-keyword ranks with reciprocal-rank fusion."""
    collection = get_collection(settings)
    where = _where_clause(source_types, domains)
    all_chunks = collection.get(where=where, include=["documents", "metadatas"])
    documents = all_chunks.get("documents", [])
    metadatas = all_chunks.get("metadatas", [])
    if not documents:
        return []
    limit = min(k or settings.retrieval_k, len(documents))
    semantic = collection.query(query_embeddings=embed_texts([query], settings), n_results=limit, where=where, include=["documents", "metadatas"])
    bm25 = BM25Okapi([_tokenize(text) for text in documents])
    scores = bm25.get_scores(_tokenize(query))
    keyword_order = sorted(range(len(documents)), key=lambda index: scores[index], reverse=True)[:limit]

    fused: dict[str, float] = defaultdict(float)
    payload: dict[str, tuple[str, dict]] = {}
    for rank, (text, metadata) in enumerate(zip(semantic["documents"][0], semantic["metadatas"][0]), start=1):
        key = f"{metadata['source']}::{metadata['chunk_index']}"
        fused[key] += 1 / (60 + rank)
        payload[key] = (text, metadata)
    for rank, index in enumerate(keyword_order, start=1):
        metadata = metadatas[index]
        key = f"{metadata['source']}::{metadata['chunk_index']}"
        fused[key] += 1 / (60 + rank)
        payload[key] = (documents[index], metadata)
    return [SearchResult(
        text=payload[key][0], title=payload[key][1].get("title", "Untitled source"),
        source=payload[key][1].get("source", ""), source_type=payload[key][1].get("source_type", "unknown"),
        chunk_index=int(payload[key][1].get("chunk_index", 0)), score=score,
    ) for key, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]]


def browse_documents(
    settings: Settings,
    query: str = "",
    source_types: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    limit: int = 50,
) -> list[SearchResult]:
    """Browse metadata and simple text matches without spending an embedding request."""
    collection = get_collection(settings)
    records = collection.get(where=_where_clause(source_types, domains), include=["documents", "metadatas"])
    needle = query.lower().strip()
    matches = []
    for text, metadata in zip(records.get("documents", []), records.get("metadatas", [])):
        title = metadata.get("title", "Untitled source")
        if needle and needle not in text.lower() and needle not in title.lower():
            continue
        matches.append(SearchResult(
            text=text, title=title, source=metadata.get("source", ""),
            source_type=metadata.get("source_type", "unknown"),
            chunk_index=int(metadata.get("chunk_index", 0)), score=1.0,
        ))
    return matches[:limit]
