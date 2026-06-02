"""
DualRAG Core — LLM Service (Google Gemini)
==========================================
Uses Google Gemini API for final answer generation.

Provides both full and streaming generation.

Synchronous methods — callers must use asyncio.to_thread(...)
or the async bridge in AnswerGenerator.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import google.generativeai as genai

from core.config import settings

logger = logging.getLogger("dualrag.llm")


class LLMService:
    def __init__(self) -> None:
        if not settings.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY is not set — Gemini calls will fail")

        genai.configure(api_key=settings.GOOGLE_API_KEY)

        self._model_name = settings.LLM_MODEL
        self._temperature = settings.LLM_TEMPERATURE
        self._max_tokens = settings.LLM_MAX_TOKENS

        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config={
                "temperature": self._temperature,
                "max_output_tokens": self._max_tokens,
            },
        )

        logger.info(
            "LLMService initialised (Gemini model=%s, temp=%.1f, max_tokens=%d)",
            self._model_name,
            self._temperature,
            self._max_tokens,
        )

    def generate(self, prompt: str, model_override: Optional[str] = None) -> str:
        model_name = model_override or self._model_name

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": self._temperature,
                "max_output_tokens": self._max_tokens,
            },
        )

        response = model.generate_content(prompt)
        text = response.text if hasattr(response, "text") else ""
        logger.debug("Generated %d chars", len(text))
        return text

    def generate_stream(
        self, prompt: str, model_override: Optional[str] = None
    ) -> Iterator[str]:
        model_name = model_override or self._model_name

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": self._temperature,
                "max_output_tokens": self._max_tokens,
            },
        )

        response = model.generate_content(prompt, stream=True)

        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text