"""Persistent semantic response cache using the configured embedding model."""

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from src.config import get_settings
from src.retrieval.embeddings import embed_texts


class SemanticCache:
    def __init__(self) -> None:
        self.path = get_settings().data_dir / "semantic_cache.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS responses (
                    query TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )""")

    def get(self, query: str) -> dict | None:
        threshold = get_settings().cache_similarity_threshold
        query_vector = embed_texts([query])[0]
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT embedding, response FROM responses"
            ).fetchall()
        for embedding_json, response_json in rows:
            cached_vector = np.asarray(json.loads(embedding_json), dtype=np.float32)
            similarity = float(np.dot(query_vector, cached_vector))
            if similarity >= threshold:
                return json.loads(response_json)
        return None

    def put(self, query: str, response: dict) -> None:
        vector = embed_texts([query])[0].tolist()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """INSERT OR REPLACE INTO responses
                   (query, embedding, response, created_at) VALUES (?, ?, ?, ?)""",
                (
                    query,
                    json.dumps(vector),
                    json.dumps(response),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def clear(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM responses")
