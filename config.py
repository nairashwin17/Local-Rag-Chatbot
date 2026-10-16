"""Central configuration for the local RAG chatbot."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseModel):
    """Runtime settings, read once from the local environment."""

    llm_provider: str = "gemini"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    chunk_size: int = Field(default=900, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_k: int = Field(default=5, ge=1, le=20)
    log_level: str = "INFO"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    docs_dir: Path = PROJECT_ROOT / "docs"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma_db"

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"openai", "gemini"}:
            raise ValueError("LLM_PROVIDER must be 'openai' or 'gemini'.")
        return value

    @property
    def active_api_key(self) -> str | None:
        return self.openai_api_key if self.llm_provider == "openai" else self.gemini_api_key

    @property
    def active_model(self) -> str:
        return self.openai_model if self.llm_provider == "openai" else self.gemini_model

    @property
    def active_embedding_model(self) -> str:
        return self.openai_embedding_model if self.llm_provider == "openai" else self.gemini_embedding_model


@lru_cache
def get_settings() -> Settings:
    """Load `.env` values and create local data folders when needed."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    for directory in (settings.raw_data_dir, settings.docs_dir, settings.chroma_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings
