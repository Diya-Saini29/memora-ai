"""
retrieval.py
------------
Hybrid retrieval over stored memories:
1. Semantic: cosine similarity between query embedding and stored memory embeddings.
2. Lexical: SQLite FTS5 keyword match as a fallback/boost (catches exact terms
   the embedding model might under-weight, e.g. proper nouns like "PULSE" or "SBI").
3. Merge: combine + re-rank by a weighted score, then trim to top_k.

This is intentionally simple (linear scan over embeddings) since SQLite has no
native vector index — fine for a personal memory store in the hundreds/low
thousands of triples. Documented as a known scaling limit (see README).
"""

from typing import Optional
import numpy as np

from . import memory_store
from . import embeddings as emb


def _triple_to_text(subject: str, predicate: str, object_: str) -> str:
    return f"{subject} {predicate.replace('_', ' ')} {object_}"


def semantic_search(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    query_vec = emb.embed_text(query)
    rows = memory_store.get_memories_with_embeddings(user_id)

    scored = []
    for row in rows:
        vec = emb.from_bytes(row["embedding"])
        sim = emb.cosine_similarity(query_vec, vec)
        scored.append({**row, "score": sim})

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


def keyword_search(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    results = memory_store.keyword_search(user_id, query, limit=top_k)
    for r in results:
        r["score"] = 0.5  # flat boost score for lexical matches
    return results

def hybrid_search(
    user_id: str,
    query: str,
    top_k: int = 5,
    semantic_weight: float = 0.92,
) -> list[dict]:
    """
    Merge semantic + lexical results. Semantic similarity dominates ranking;
    importance_score only acts as a light tiebreaker between close matches,
    not a competing factor that can bury a highly relevant but low-importance
    memory (e.g. a hobby) behind an unrelated high-importance one.
    """
    semantic_results = semantic_search(user_id, query, top_k=top_k * 2)
    lexical_results = keyword_search(user_id, query, top_k=top_k * 2)

    merged: dict[str, dict] = {}

    for r in semantic_results:
        merged[r["id"]] = {
            **r,
            "final_score": r["score"] * semantic_weight
            + r.get("importance_score", 0.5) * (1 - semantic_weight),
        }

    for r in lexical_results:
        if r["id"] in merged:
            merged[r["id"]]["final_score"] += 0.15
        else:
            merged[r["id"]] = {
                **r,
                "final_score": 0.4 + r.get("importance_score", 0.5) * 0.1,
            }

    ranked = sorted(merged.values(), key=lambda r: r["final_score"], reverse=True)
    return ranked[:top_k]

def format_memories_for_context(memories: list[dict]) -> str:
    """Turn retrieved memory rows into a compact context block for the LLM prompt."""
    if not memories:
        return ""
    lines = [
        f"- {m['subject']} {m['predicate'].replace('_', ' ')} {m['object']}"
        for m in memories
    ]
    return "Known facts about the user:\n" + "\n".join(lines)
