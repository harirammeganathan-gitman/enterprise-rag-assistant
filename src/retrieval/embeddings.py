"""Lazy shared sentence-transformer embedding model."""

from functools import lru_cache

import numpy as np

from src.config import get_settings


@lru_cache
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    return get_embedding_model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
