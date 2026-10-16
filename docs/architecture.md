# Local RAG Chatbot Architecture

Phase 1 is a local, single-process retrieval-augmented generation application. It is intentionally small enough to read end to end.

## Data flows

```text
docs/ and data/raw/ -> document loader -> text splitter -> provider embeddings -> ChromaDB

Streamlit question -> hybrid retriever -> ChromaDB semantic search + BM25 keyword search
                   -> rank fusion -> evidence-bound LLM prompt -> answer + citations
```

`ingestion/` handles only document collection and indexing. `rag/retriever.py` finds evidence; `rag/answer.py` turns supplied evidence into an answer. This separation makes retrieval and answer behavior independently testable.

## Design decisions

- **ChromaDB** persists local vectors under `data/chroma_db/`; the folder is ignored by Git.
- **Source fingerprints** make reruns incremental: unchanged sources are skipped and a changed source replaces its former chunks.
- **Handbook-first collection** treats the approved document in `docs/` as the source of truth. The first version does not fetch website or GitHub content.
- **Hybrid retrieval** combines semantic similarity with BM25 keyword ranks, which helps with names, filenames, and technical terms.
- **Provider adapters** use the OpenAI and Google GenAI SDKs directly. Set `LLM_PROVIDER` in `.env` to choose one.
- **No evidence, no answer:** if retrieval returns no chunks, the application explicitly says it lacks indexed evidence.

## Phase 1 boundaries

The project includes knowledge-domain/source-type filters, a Knowledge Library, and source refresh behavior. Add approved PDFs or other local documents to `data/raw/`; website ingestion and Localization QA remain later work.
