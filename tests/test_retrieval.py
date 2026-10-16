from types import SimpleNamespace

from config import Settings, get_settings
from rag.answer import generate_answer
from rag.embeddings import embed_texts
from rag.retriever import SearchResult, _where_clause


def test_empty_context_reports_insufficient_evidence() -> None:
    answer = generate_answer("What is not indexed?", [], get_settings())
    assert answer.insufficient_evidence is True
    assert answer.citations == []


def test_answer_model_has_citations_without_calling_provider(monkeypatch) -> None:
    monkeypatch.setattr("rag.answer.generate_text", lambda prompt, settings: "Curious Learning supports literacy [1].")
    result = SearchResult("Evidence", "Handbook", "handbook.md", "md", 1.0)
    answer = generate_answer("What is the mission?", [result], get_settings())
    assert answer.citations[0].title == "Handbook"
    assert "[1]" in answer.answer


def test_gemini_embedding_keeps_client_open_for_request(monkeypatch) -> None:
    class FakeClient:
        latest: "FakeClient"

        def __init__(self, api_key: str) -> None:
            self.closed = False
            self.models = self
            FakeClient.latest = self

        def embed_content(self, **kwargs):
            assert self.closed is False
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("google.genai.Client", FakeClient)
    settings = Settings(llm_provider="gemini", gemini_api_key="test-key")
    assert embed_texts(["a document"], settings) == [[0.1, 0.2]]
    assert FakeClient.latest.closed is True


def test_retrieval_filters_build_chroma_where_clause() -> None:
    assert _where_clause(["website"], None) == {"source_type": {"$in": ["website"]}}
    assert _where_clause(["website"], ["research"]) == {
        "$and": [{"source_type": {"$in": ["website"]}}, {"knowledge_domain": {"$in": ["research"]}}]
    }
