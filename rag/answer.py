"""Evidence-bound answer generation and citation formatting."""

from __future__ import annotations

from pydantic import BaseModel

from config import Settings
from rag.embeddings import generate_text
from rag.prompts import ANSWER_PROMPT
from rag.retriever import SearchResult


class Citation(BaseModel):
    number: int
    title: str
    source: str
    excerpt: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool


def generate_answer(question: str, context: list[SearchResult], settings: Settings) -> Answer:
    """Answer only with retrieved evidence, or report that no evidence was found."""
    if not context:
        return Answer(answer="I don't have enough indexed evidence to answer that question yet.", citations=[], insufficient_evidence=True)
    evidence = "\n\n".join(
        f"[{number}] {item.title}\nSource: {item.source}\n{item.text}"
        for number, item in enumerate(context, start=1)
    )
    response = generate_text(ANSWER_PROMPT.format(question=question, evidence=evidence), settings)
    citations = [Citation(number=index, title=item.title, source=item.source, excerpt=item.text[:280].strip()) for index, item in enumerate(context, start=1)]
    return Answer(answer=response, citations=citations, insufficient_evidence=False)
