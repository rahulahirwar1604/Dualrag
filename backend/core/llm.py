from __future__ import annotations

import logging
from typing import Iterator, Optional

from google import genai

from core.config import settings

logger = logging.getLogger("dualrag.llm")


class LLMService:

    def __init__(self) -> None:

        self.client = genai.Client(
            api_key=settings.GOOGLE_API_KEY
        )

        self._model_name = settings.LLM_MODEL

        logger.info(
            "LLMService initialised (Gemini model=%s)",
            self._model_name,
        )

    def generate(
        self,
        prompt: str,
        model_override: Optional[str] = None,
    ) -> str:

        model_name = model_override or self._model_name

        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        return response.text

    def generate_stream(
        self,
        prompt: str,
        model_override: Optional[str] = None,
    ) -> Iterator[str]:

        model_name = model_override or self._model_name

        for chunk in self.client.models.generate_content_stream(
            model=model_name,
            contents=prompt,
        ):
            if chunk.text:
                yield chunk.text