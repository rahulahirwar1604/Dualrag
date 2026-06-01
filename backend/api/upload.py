"""
DualRAG API — Document Upload
==============================
POST /api/upload

Thin controller: validates input, reads file bytes, delegates to
``IngestionService``, and formats the HTTP response.

Frontend contract:
  - FormData field name: ``file``
  - Response: { message, document_id, filename, chunks }
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from services.ingestion import IngestionService

logger = logging.getLogger("dualrag.api.upload")

router = APIRouter(tags=["upload"])

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Module-level service instance (stateless — safe to share)
_ingestion = IngestionService()


@router.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload and index a document.

    Validates the file type, reads bytes asynchronously, then delegates
    the entire parse → chunk → embed → upsert → persist pipeline to
    ``IngestionService.ingest()`` via ``asyncio.to_thread()``.
    """
    # -- Access singletons from app.state --
    vector_store = request.app.state.vector_store
    embedding_service = request.app.state.embedding_service

    # 1. Validate extension
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Read file bytes (async I/O)
    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    logger.info("Received upload: '%s' (%d bytes)", filename, len(file_bytes))

    # 3. Delegate to ingestion service (sync → offload to thread)
    try:
        result = await asyncio.to_thread(
            _ingestion.ingest,
            file_bytes=file_bytes,
            filename=filename,
            extension=ext,
            vector_store=vector_store,
            embedding_service=embedding_service,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return result
