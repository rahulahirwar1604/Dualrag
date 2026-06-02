"""
DualRAG Core — Simple Local Reranker
"""

from typing import Any, Dict, List


class RerankService:
    def __init__(self):
        pass

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:

        for chunk in chunks:
            chunk["relevance_score"] = chunk.get("score", 0.0)

        return chunks[:top_n]