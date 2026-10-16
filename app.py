"""Streamlit entry point for the local RAG chatbot."""

from __future__ import annotations

import streamlit as st

from config import get_settings
from rag.answer import generate_answer
from rag.embeddings import ProviderConfigurationError
from rag.retriever import browse_documents, get_collection, search_documents

st.set_page_config(page_title="Local RAG Chatbot", page_icon="📚", layout="wide")
settings = get_settings()
st.title("Local RAG Chatbot")
st.caption("Answers are based only on documents indexed in this local knowledge library.")

page = st.sidebar.radio("Workspace", ["Ask your documents", "Knowledge Library"])


def _library_filters() -> tuple[list[str], list[str]]:
    """Populate only filters that are currently represented in the index."""
    records = get_collection(settings).get(include=["metadatas"])
    metadata = records.get("metadatas", [])
    types = sorted({item.get("source_type", "unknown") for item in metadata})
    domains = sorted({item.get("knowledge_domain", "unknown") for item in metadata})
    selected_types = st.multiselect("Source type", types, default=types)
    selected_domains = st.multiselect("Knowledge domain", domains, default=domains)
    return selected_types, selected_domains


if page == "Knowledge Library":
    st.subheader("Knowledge Library")
    st.caption("Browse the indexed evidence. To refresh it, rerun `python -m ingestion.index_documents`.")
    source_types, domains = _library_filters()
    term = st.text_input("Search title or document text")
    results = browse_documents(settings, term, source_types, domains)
    st.write(f"{len(results)} matching chunks")
    for result in results:
        with st.expander(f"{result.title} · {result.source_type}"):
            st.caption(result.source)
            st.write(result.text[:1000])
            if result.source.startswith("http"):
                st.link_button("Open original source", result.source)
elif "messages" not in st.session_state:
    st.session_state.messages = []

if page == "Ask your documents":
    with st.sidebar:
        st.caption("Narrow the evidence searched for each answer.")
        source_types, domains = _library_filters()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about your indexed documents")
else:
    question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            results = search_documents(question, settings, source_types=source_types, domains=domains)
            answer = generate_answer(question, results, settings)
            st.markdown(answer.answer)
            if answer.citations:
                with st.expander("Sources"):
                    for citation in answer.citations:
                        st.markdown(f"**[{citation.number}] {citation.title}**  ")
                        st.caption(citation.source)
                        st.write(citation.excerpt)
                        if citation.source.startswith("http"):
                            st.link_button("Open original source", citation.source, key=f"citation-{citation.number}")
            st.session_state.messages.append({"role": "assistant", "content": answer.answer})
        except ProviderConfigurationError as error:
            st.info(f"{error} Then run `python -m ingestion.index_documents` before asking questions.")
        except Exception as error:
            st.error(f"Could not answer the question: {error}")
