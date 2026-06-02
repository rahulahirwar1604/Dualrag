"""
DualRAG Core — Embedding Service (OpenRouter Raw HTTP)
======================================================
Uses direct HTTP requests to OpenRouter embeddings endpoint for
nvidia/llama-nemotron-embed-vl-1b-v2:free.
"""

from __future__ import annotations

import logging
from typing import List

import google.generativeai as genai

from core.config import settings

logger = logging.getLogger("dualrag.embeddings")


class EmbeddingService:

    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)

        self.model = "text-embedding-004"

    def embed_query(self, text: str) -> List[float]:

        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )

        return result["embedding"]

    def embed_texts(self, texts: List[str]):

        vectors = []

        for text in texts:

            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )

            vectors.append(result["embedding"])

        return vectors
