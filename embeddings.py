"""
embeddings.py
-------------
Local, free embedding generation using sentence-transformers (all-MiniLM-L6-v2).

This model is small (~80MB), runs fine on CPU, and produces 384-dim vectors
that are good enough for semantic similarity over a personal memory store
(hundreds to low-thousands of triples). No API key, no network calls after
the first model download.
"""

import numpy as np
from functools import lru_cache

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    # device="cpu" forced explicitly — recent torch/sentence-transformers
    # combos on Windows can otherwise try to init on a "meta" device and
    # crash with "Cannot copy out of meta tensor; no data!"
    return SentenceTransformer(_MODEL_NAME, device="cpu")


def embed_text(text: str) -> np.ndarray:
    """Return a 384-dim float32 embedding for a single string."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Return an (N, 384) float32 matrix for a list of strings."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return np.asarray(vecs, dtype=np.float32)


def to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already normalized at encode time, so dot product == cosine sim.
    # Still guard against zero vectors defensively.
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
