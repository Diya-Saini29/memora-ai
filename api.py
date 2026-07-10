"""
api.py
------
FastAPI server for Memora.

Endpoints:
- POST /chat            : send a user message, get an assistant reply (with memory context injected)
- GET  /memories/{user_id}   : list a user's memories (for the dashboard)
- PUT  /memories/{memory_id} : edit a memory
- DELETE /memories/{memory_id} : soft-delete a memory
- POST /extract/{user_id}/{conversation_id} : manually trigger extraction (also runs automatically)

Extraction trigger: every BATCH_SIZE messages in a conversation, we run extraction
over the most recent unprocessed batch. Kept simple (message-count based) rather
than a background task queue, since this is a single-user local demo.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from . import memory_store
from . import extraction
from . import retrieval
from . import embeddings as emb

load_dotenv()

BATCH_SIZE = int(os.getenv("EXTRACTION_BATCH_SIZE", "18"))

app = FastAPI(title="Memora API")


@app.on_event("startup")
def startup():
    memory_store.init_db()


class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    retrieved_memories: list[dict]
    extraction_triggered: bool


class MemoryUpdateRequest(BaseModel):
    subject: str
    predicate: str
    object: str


def _generate_reply(user_message: str, context_block: str) -> str:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — see .env.example")
    genai.configure(api_key=api_key)

    system_prompt = (
        "You are a helpful assistant with long-term memory about the user. "
        "Use the known facts below naturally, only when relevant — don't force them in.\n\n"
        f"{context_block}"
    )
    model = genai.GenerativeModel(model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"), system_instruction=system_prompt)
    try:
        response = model.generate_content(user_message)
        return response.text
    except ResourceExhausted:
        return "⚠️ Daily Gemini free-tier quota reached — try again after quota reset (midnight Pacific Time)."


def _run_extraction_if_due(user_id: str, conversation_id: str) -> bool:
    messages = memory_store.get_conversation_messages(user_id, conversation_id, limit=1000)
    if len(messages) == 0 or len(messages) % BATCH_SIZE != 0:
        return False

    batch = messages[-BATCH_SIZE:]
    triples = extraction.extract_triples(
        [{"role": m["role"], "content": m["content"]} for m in batch]
    )

    for t in triples:
        is_multi_valued = t["predicate"] in memory_store.MULTI_VALUED_PREDICATES
        if is_multi_valued:
            existing = memory_store.find_exact_memory(user_id, t["subject"], t["predicate"], t["object"])
        else:
            existing = memory_store.find_similar_memory(user_id, t["subject"], t["predicate"])
        if existing:
            if not is_multi_valued:
                memory_store.update_memory(existing["id"], t["subject"], t["predicate"], t["object"])
            mem_id = existing["id"]
        else:
            mem_id = memory_store.add_memory(
                user_id=user_id,
                subject=t["subject"],
                predicate=t["predicate"],
                object_=t["object"],
                confidence=t["confidence"],
                importance_score=t["importance"],
            )
            text_content = retrieval._triple_to_text(t["subject"], t["predicate"], t["object"])
            vec = emb.embed_text(text_content)
            memory_store.add_memory_embedding(mem_id, emb.to_bytes(vec), text_content)
    return len(triples) > 0


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    memory_store.add_message(req.user_id, req.conversation_id, "user", req.message)

    retrieved = retrieval.hybrid_search(req.user_id, req.message, top_k=5)
    context_block = retrieval.format_memories_for_context(retrieved)

    reply = _generate_reply(req.message, context_block)
    memory_store.add_message(req.user_id, req.conversation_id, "assistant", reply)

    extraction_triggered = _run_extraction_if_due(req.user_id, req.conversation_id)

    return ChatResponse(
        reply=reply,
        retrieved_memories=[
            {"subject": m["subject"], "predicate": m["predicate"], "object": m["object"], "score": m.get("score", m.get("final_score", 0))}
            for m in retrieved
        ],
        extraction_triggered=extraction_triggered,
    )
def find_similar_memory(user_id: str, subject: str, predicate: str):
    """Check for an existing memory with the same subject+predicate before inserting
    a new one, so repeated facts update in place instead of piling up as duplicates."""
    conn = memory_store.get_connection()
    row = conn.execute(
        """SELECT * FROM memories
           WHERE user_id = ? AND subject = ? AND predicate = ? AND is_deleted = 0
           ORDER BY created_at DESC LIMIT 1""",
        (user_id, subject, predicate),
    ).fetchone()
    conn.close()
    return dict(row) if row else None

@app.get("/memories/{user_id}")
def list_memories(user_id: str):
    return memory_store.get_all_memories(user_id)


@app.put("/memories/{memory_id}")
def edit_memory(memory_id: str, req: MemoryUpdateRequest):
    memory_store.update_memory(memory_id, req.subject, req.predicate, req.object)
    return {"status": "updated"}


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str):
    memory_store.soft_delete_memory(memory_id)
    return {"status": "deleted"}


@app.post("/extract/{user_id}/{conversation_id}")
def manual_extract(user_id: str, conversation_id: str):
    messages = memory_store.get_conversation_messages(user_id, conversation_id, limit=1000)
    triples = extraction.extract_triples(
        [{"role": m["role"], "content": m["content"]} for m in messages]
    )
    created = []
    for t in triples:
        is_multi_valued = t["predicate"] in memory_store.MULTI_VALUED_PREDICATES
        if is_multi_valued:
            existing = memory_store.find_exact_memory(user_id, t["subject"], t["predicate"], t["object"])
        else:
            existing = memory_store.find_similar_memory(user_id, t["subject"], t["predicate"])
        if existing:
            if not is_multi_valued:
                memory_store.update_memory(existing["id"], t["subject"], t["predicate"], t["object"])
            mem_id = existing["id"]
        else:
            mem_id = memory_store.add_memory(
                user_id=user_id,
                subject=t["subject"],
                predicate=t["predicate"],
                object_=t["object"],
                confidence=t["confidence"],
                importance_score=t["importance"],
            )
            text_content = retrieval._triple_to_text(t["subject"], t["predicate"], t["object"])
            vec = emb.embed_text(text_content)
            memory_store.add_memory_embedding(mem_id, emb.to_bytes(vec), text_content)
        created.append(mem_id)
    return {"created_memory_ids": created, "count": len(created)}

@app.get("/debug/retrieve/{user_id}")
def debug_retrieve(user_id: str, query: str, top_k: int = 5):
    """Test retrieval quality without spending Gemini quota on a chat reply."""
    results = retrieval.hybrid_search(user_id, query, top_k=top_k)
    return [
        {
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "final_score": round(r.get("final_score", r.get("score", 0)), 3),
        }
        for r in results
    ]