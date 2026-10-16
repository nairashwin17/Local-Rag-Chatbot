from pathlib import Path

from ingestion.load_documents import load_local_documents
from ingestion.index_documents import _chunk_id
from ingestion.split_documents import split_documents


def test_load_and_split_preserves_source_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text("Curious Learning supports literacy. " * 40, encoding="utf-8")
    documents = load_local_documents(tmp_path)
    chunks = split_documents(documents, chunk_size=80, chunk_overlap=10)
    assert documents[0].metadata["title"] == "sample"
    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == str(source.resolve()) for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_chunk_id_is_stable_for_duplicate_prevention() -> None:
    assert _chunk_id("source.md", 0, "same text") == _chunk_id("source.md", 0, "same text")
    assert _chunk_id("source.md", 0, "same text") != _chunk_id("source.md", 1, "same text")
