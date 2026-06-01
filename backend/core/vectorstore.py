"""
DualRAG Core — Qdrant Vector Store Manager
============================================
Handles Qdrant collection lifecycle and provides low-level vector CRUD
operations.  Higher-level retrieval logic lives in services/retrieval.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from core.config import settings

logger = logging.getLogger("dualrag.vectorstore")


class VectorStoreManager:
    """Manages the Qdrant collection used by DualRAG."""

    def __init__(self) -> None:
        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=30,
        )
        self._collection = settings.QDRANT_COLLECTION

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist, then
        ensure a keyword payload index on ``document_id`` for efficient
        filtered deletes and searches."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", self._collection)
        else:
            logger.info("Qdrant collection '%s' already exists", self._collection)

        # Create payload index on document_id for fast filtered ops.
        # This is idempotent — Qdrant ignores if the index already exists.
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Payload index on 'document_id' ensured")

    # ------------------------------------------------------------------
    # Upsert vectors
    # ------------------------------------------------------------------
    def upsert_chunks(
        self,
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> int:
        """
        Insert chunk vectors with metadata payloads.

        Parameters
        ----------
        embeddings : list of float vectors
        payloads : list of dicts, each containing:
            - document_id
            - filename
            - chunk_id
            - chunk_text
            - upload_timestamp

        Returns
        -------
        int — number of points upserted
        """
        points = [
            PointStruct(
                id=str(uuid4()),
                vector=emb,
                payload=payload,
            )
            for emb, payload in zip(embeddings, payloads)
        ]

        # Qdrant recommends batching ≤ 100 points per upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )

        logger.info(
            "Upserted %d vectors for document '%s'",
            len(points),
            payloads[0].get("filename", "unknown") if payloads else "unknown",
        )
        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: List[float],
        top_k: int = 15,
        doc_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Returns list of dicts with keys: score, document_id, filename,
        chunk_id, chunk_text.
        """
        search_filter = None
        if doc_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=doc_filter),
                    )
                ]
            )

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        return [
            {
                "score": hit.score,
                "document_id": hit.payload.get("document_id", ""),
                "filename": hit.payload.get("filename", ""),
                "chunk_id": hit.payload.get("chunk_id", ""),
                "chunk_text": hit.payload.get("chunk_text", ""),
            }
            for hit in results
        ]

    # ------------------------------------------------------------------
    # Delete by document_id
    # ------------------------------------------------------------------
    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all vectors associated with a given document_id."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info("Deleted all vectors for document_id=%s", document_id)

    # ------------------------------------------------------------------
    # Collection stats
    # ------------------------------------------------------------------
    def collection_info(self) -> Dict[str, Any]:
        """Return basic stats about the collection."""
        info = self._client.get_collection(self._collection)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value if info.status else "unknown",
        }
