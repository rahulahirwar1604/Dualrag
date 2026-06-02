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

        self.model = settings.EMBEDDING_MODEL

        logger.info(
            "EmbeddingService initialised (model=%s)",
            self.model,
        )

    def embed_query(self, text: str) -> List[float]:

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )

        return response.embeddings[0].values

    def embed_texts(self, texts: List[str]) -> List[List[float]]:

        vectors = []

        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
            )

            vectors.append(
                response.embeddings[0].values
            )

        return vectors