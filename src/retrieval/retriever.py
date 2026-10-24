"""Hybrid dense/BM25 retrieval, reciprocal-rank fusion, and cross-encoder reranking."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

from src.config import get_settings
from src.retrieval.embeddings import embed_texts
from src.retrieval.index import _client, load_corpus

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@lru_cache
def _reranker() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


class HybridRetriever:
    """Retrieve and rerank matching document chunks."""

    def __init__(self, client: QdrantClient | None = None) -> None:
        self.client = client or _client()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        settings = get_settings()
        corpus = load_corpus()
        if not corpus:
            return []
        candidates = settings.retrieval_candidates
        from rank_bm25 import BM25Okapi

        dense_results = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=embed_texts([query])[0].tolist(),
            limit=candidates,
            with_payload=True,
        ).points
        dense_rank = {}
        for rank, point in enumerate(dense_results, start=1):
            payload = point.payload or {}
            key = f"{payload.get('file_hash')}:{payload.get('chunk_index')}"
            dense_rank[key] = (rank, payload)

        bm25 = BM25Okapi([_tokens(chunk.text) for chunk in corpus])
        sparse_scores = bm25.get_scores(_tokens(query))
        sparse_order = sorted(
            range(len(corpus)), key=lambda index: sparse_scores[index], reverse=True
        )[:candidates]
        sparse_rank: dict[str, tuple[int, dict]] = {}
        for rank, index in enumerate(sparse_order, start=1):
            chunk = corpus[index]
            sparse_rank[f"{chunk.file_hash}:{chunk.chunk_index}"] = (
                rank,
                chunk.model_dump(),
            )

        fused: dict[str, tuple[float, dict]] = {}
        for ranked in (dense_rank, sparse_rank):
            for key, (rank, payload) in ranked.items():
                score, _ = fused.get(key, (0.0, payload))
                fused[key] = (score + 1 / (60 + rank), payload)
        ordered = sorted(fused.values(), key=lambda item: item[0], reverse=True)
        shortlist = [payload for _, payload in ordered[:candidates]]
        if not shortlist:
            return []

        rerank_scores = _reranker().predict(
            [[query, item["text"]] for item in shortlist]
        )
        reranked = sorted(
            zip(shortlist, rerank_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        result_count = settings.top_k if top_k is None else top_k
        return [
            {**payload, "rerank_score": float(score)}
            for payload, score in reranked[:result_count]
        ]
