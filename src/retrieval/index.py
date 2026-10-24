"""Qdrant vector index and local persistent corpus for BM25."""

import json
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from src.config import get_settings
from src.ingestion.models import DocumentChunk
from src.retrieval.embeddings import embed_texts


@lru_cache
def _client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def index_chunks(chunks: list[DocumentChunk]) -> int:
    """Upsert chunk vectors and atomically replace the BM25 source corpus."""
    if not chunks:
        return 0
    settings = get_settings()
    client = _client()
    vectors = embed_texts([chunk.text for chunk in chunks])
    collection = settings.qdrant_collection
    existing = {item.name for item in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )

    points = []
    for chunk, vector in zip(chunks, vectors):
        key = f"{chunk.file_hash}:{chunk.chunk_index}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
        payload = chunk.model_dump()
        points.append(
            models.PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
        )
    client.upsert(collection_name=collection, points=points, wait=True)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = settings.data_dir / "indexed_chunks.jsonl"
    temporary_path = corpus_path.with_suffix(".jsonl.tmp")
    merged = {(chunk.file_hash, chunk.chunk_index): chunk for chunk in load_corpus()}
    merged.update({(chunk.file_hash, chunk.chunk_index): chunk for chunk in chunks})
    with temporary_path.open("w", encoding="utf-8") as stream:
        for chunk in merged.values():
            stream.write(chunk.model_dump_json() + "\n")
    temporary_path.replace(corpus_path)
    return len(chunks)


def load_corpus() -> list[DocumentChunk]:
    path = get_settings().data_dir / "indexed_chunks.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [
            DocumentChunk.model_validate(json.loads(line))
            for line in stream
            if line.strip()
        ]
