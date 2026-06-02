"""
DualRAG Core — Embedding Service (Gemini)
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

        # IMPORTANT
        self.model = settings.EMBEDDING_MODEL

        logger.info(
            "EmbeddingService initialised (model=%s)",
            self.model,
        )

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query",
        )

        return result["embedding"]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = []

        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
            )

            vectors.append(result["embedding"])

        return vectors