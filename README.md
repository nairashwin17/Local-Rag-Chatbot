# Local RAG Chatbot

A deliberately simple local knowledge assistant. It indexes approved documents, retrieves evidence with ChromaDB and BM25, and answers questions in Streamlit with source citations.

## What Phase 1 includes

- Local Markdown, text, PDF, and DOCX ingestion.
- Persistent ChromaDB vector index and BM25 keyword retrieval.
- OpenAI or Gemini configuration through `.env`.
- Evidence-only answers with source titles, paths, and excerpts.
- A beginner-friendly Streamlit chat and Knowledge Library with source-type and domain filters.
- Unit tests that do not call paid model APIs.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set either `OPENAI_API_KEY` (the default provider) or set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`.

## Run the first knowledge base

Documents in `docs/` are indexed by default. Put any additional approved local files in `data/raw/`; project architecture notes are intentionally excluded. No website or GitHub content is fetched by the ingestion command.

```bash
python -m ingestion.index_documents
python -m streamlit run app.py
```

Rerun the ingestion command after adding or editing a document. Unchanged chunks keep their stable IDs and are not duplicated.

## Test

```bash
pytest
```

## Project map

```text
ingestion/  Collect, extract, split, and index documents
rag/        Provider adapters, hybrid retrieval, prompts, and answers
tools/      Small deterministic validation helpers for Localization QA
data/       Local-only source data, processed artifacts, and ChromaDB
tests/      Fast unit tests with external calls mocked
```

Read `docs/architecture.md` for the data flow and deliberately limited Phase 1 scope.
