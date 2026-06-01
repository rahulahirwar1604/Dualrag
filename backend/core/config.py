"""
DualRAG Core — Application Configuration
==========================================
Centralised settings loaded from environment variables via pydantic-settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Server ─────────────────────────────────────
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)

    # ── CORS ───────────────────────────────────────
    CORS_ORIGINS: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173"
    )

    # ── Optional Security ──────────────────────────
    API_KEY: str = Field(default="")

    # ── OpenRouter (Embeddings only) ──────────────
    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")

    EMBEDDING_MODEL: str = Field(default="nvidia/llama-nemotron-embed-vl-1b-v2:free")
    EMBEDDING_DIMENSIONS: int = Field(default=2048)

    # ── Google Gemini Final LLM ───────────────────
    GOOGLE_API_KEY: str = Field(default="")

    LLM_MODEL: str = Field(default="gemini-2.5-flash")
    LLM_TEMPERATURE: float = Field(default=0.2)
    LLM_MAX_TOKENS: int = Field(default=2048)

    # ── NVIDIA BUILD Reranker ─────────────────────
    NVIDIA_API_KEY: str = Field(default="")
    NVIDIA_RERANK_URL: str = Field(default="https://ai.api.nvidia.com/v1/ranking")
    RERANK_MODEL: str = Field(default="nvidia/llama-nemotron-rerank-vl-1b-v2")
    RERANK_TOP_N: int = Field(default=5)

    # ── Qdrant ─────────────────────────────────────
    QDRANT_URL: str = Field(default="")
    QDRANT_API_KEY: str = Field(default="")
    QDRANT_COLLECTION: str = Field(default="dualrag_documents")

    # ── Chunking ───────────────────────────────────
    CHUNK_SIZE: int = Field(default=800)
    CHUNK_OVERLAP: int = Field(default=150)

    # ── Retrieval ──────────────────────────────────
    RETRIEVAL_TOP_K: int = Field(default=15)

    # ── Storage ────────────────────────────────────
    DOCUMENT_STORE_PATH: str = Field(default="storage/documents.json")

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    

    @property
    def document_store_absolute_path(self) -> Path:
        p = Path(self.DOCUMENT_STORE_PATH)
        if p.is_absolute():
            return p
        return Path(__file__).resolve().parent.parent / p

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


settings = Settings()