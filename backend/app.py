"""
DualRAG Backend — FastAPI Application Entry Point
"""

import sys
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure backend package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from core.config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.DEBUG if settings.DEBUG else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-24s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("dualrag")


# ---------------------------------------------------------------------------
# Document Storage
# ---------------------------------------------------------------------------
def _init_document_storage() -> None:
    store_path = Path(settings.DOCUMENT_STORE_PATH)

    store_path.parent.mkdir(parents=True, exist_ok=True)

    if not store_path.exists():
        store_path.write_text(
            json.dumps([], indent=2),
            encoding="utf-8",
        )
        logger.info("Created document storage at %s", store_path)
        return

    try:
        data = json.loads(
            store_path.read_text(encoding="utf-8")
        )

        if not isinstance(data, list):
            raise ValueError("documents.json root must be a list")

        logger.info(
            "Loaded document storage with %d document(s)",
            len(data),
        )

    except Exception as exc:
        logger.warning(
            "Invalid document storage (%s). Resetting.",
            exc,
        )

        store_path.write_text(
            json.dumps([], indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    from core.vectorstore import VectorStoreManager
    from core.memory import ConversationMemory
    from core.embeddings import EmbeddingService

    from services.generator import AnswerGenerator
    from services.reranker import RerankService

    logger.info("=" * 60)
    logger.info("DualRAG Backend starting up …")
    logger.info("=" * 60)

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------
    _init_document_storage()

    # --------------------------------------------------
    # Services
    # --------------------------------------------------
    vector_store = VectorStoreManager()

    memory = ConversationMemory(max_turns=3)

    embedding_service = EmbeddingService()

    reranker = RerankService()

    generator = AnswerGenerator()

    # --------------------------------------------------
    # App State
    # --------------------------------------------------
    app.state.vector_store = vector_store
    app.state.memory = memory
    app.state.embedding_service = embedding_service
    app.state.reranker = reranker
    app.state.generator = generator

    # --------------------------------------------------
    # Qdrant
    # --------------------------------------------------
    try:
        vector_store.ensure_collection()

        logger.info(
            "Qdrant collection '%s' ready",
            settings.QDRANT_COLLECTION,
        )

    except Exception as exc:
        logger.exception(
            "Failed to initialize Qdrant: %s",
            exc,
        )

    logger.info(
        "DualRAG Backend is ready — accepting requests"
    )

    logger.info(
        "Expecting frontend at origins: %s",
        ", ".join(settings.cors_origins_list),
    )

    yield

    logger.info("DualRAG Backend shutting down …")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DualRAG API",
    description="Agentic Intelligence — Document Chat RAG Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
from api.health import router as health_router
from api.upload import router as upload_router
from api.documents import router as documents_router
from api.query import router as query_router

app.include_router(health_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(query_router, prefix="/api")


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "DualRAG API",
        "version": "1.0.0",
        "health": "/api/health",
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )