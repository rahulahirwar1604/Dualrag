"""
DualRAG Core — Embedding Service (OpenRouter Raw HTTP)
======================================================
Uses direct HTTP requests to OpenRouter embeddings endpoint for
nvidia/llama-nemotron-embed-vl-1b-v2:free.
"""

from __future__ import annotations

import logging
from typing import List

from google import genai

from core.config import settings

logger = logging.getLogger("dualrag.embeddings")


class EmbeddingService:

    def __init__(self):

        self.client = genai.Client(
            api_key=settings.GOOGLE_API_KEY
        )

        self.model = "text-embedding-004"

    def embed_query(self, text: str) -> List[float]:

        result = self.client.models.embed_content(
            model=self.model,
            contents=text
        )

        return result.embeddings[0].values

    def embed_texts(self, texts: List[str]):

        vectors = []

        for text in texts:

            result = self.client.models.embed_content(
                model=self.model,
                contents=text
            )

            vectors.append(
                result.embeddings[0].values
            )

        return vectors