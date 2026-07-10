# Memora

A personalized AI memory layer that extracts structured knowledge-graph facts (subject–predicate–object triples) from conversations, stores them with semantic embeddings, and retrieves them via hybrid search to personalize future responses.

**Cost: $0.00 (free-tier APIs + local models).** Extraction runs on Gemini's free tier, embeddings run locally, storage is SQLite.

## Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a Gemini API Key
- Visit https://aistudio.google.com/apikey
- Create a free key
- Copy `.env.example` to `.env` and paste your key

### 3. Run the System
```bash
# Terminal 1: Start backend
python -m uvicorn backend.api:app --reload --port 8000

# Terminal 2: Start frontend
streamlit run frontend/app.py
```

Open `http://localhost:8501` and start chatting.

## Architecture
User Message
↓
Hybrid Retrieval (Semantic + Lexical)
↓
Retrieved Memories (top-5) + Message History
↓
Gemini Chat with Context
↓
Chat Response + Memory Extraction Trigger
↓
Every 18 messages: Extract Triples → Embed Locally → Store in SQLite

### Core Components

| File | Purpose |
|---|---|
| `backend/memory_store.py` | SQLite schema + CRUD for triples, messages, embeddings |
| `backend/embeddings.py` | Local embedding generation (all-MiniLM-L6-v2, 384-dim) |
| `backend/extraction.py` | Gemini-based extraction of (subject, predicate, object) triples |
| `backend/retrieval.py` | Hybrid semantic + lexical search with importance weighting |
| `backend/api.py` | FastAPI endpoints: chat, extraction, memory CRUD |
| `frontend/app.py` | Streamlit UI: chat interface + memory dashboard |

## How It Works

**Extraction** — Batches of recent messages are sent to Gemini with a structured prompt:
Extract facts as JSON: [{"subject": "user", "predicate": "prefers", "object": "Python", ...}]
Response is parsed, validated, and stored as triples.

**Storage** — Triples are stored in SQLite with:
- Semantic embeddings (all-MiniLM-L6-v2) serialized as BLOB
- FTS5 index for lexical fallback
- Soft-delete + dedup for multi-valued predicates

**Retrieval** — Query triggers:
1. Semantic search: cosine similarity of embeddings
2. Lexical search: FTS5 keyword match
3. Merge & rank: importance score acts as tiebreaker, not primary ranker
4. Return top-5 to inject into chat context

**Dedup** — Multi-valued predicates (e.g., `career_goal`, `enjoys`) create new rows for distinct values; single-valued predicates (e.g., `academic_year`) update in place.

## Known Limitations

**Free-Tier API Instability**
- Model deprecation without warning (gemini-2.5-flash-lite was deprecated overnight)
- Daily quota is tight (20-30 requests/day depending on account age)
- Production would need paid tier or multi-provider fallback

**Retrieval at Scale**
- SQLite has no native vector index; retrieval does full linear scan of embeddings
- Fine up to ~1000-5000 memories; beyond that needs pgvector/Qdrant/Pinecone
- Would require ~5ms per query at 1M memories vs <1ms with real vector DB

**Semantic Search Limitations**
- Embeddings excel at exact-term matching but can blur conceptual categories
- Example: "Do I have hobbies?" may rank career goals similarly without predicate hints
- Workaround: controlled predicate vocabulary in extraction

**No Conflict Resolution**
- If user contradicts an earlier fact, both triples persist
- V2 would detect + resolve contradictions or flag for user review

**Single-User Local Demo**
- No auth, no multi-tenant isolation
- Perfect for portfolio demo; not production-ready

## Evaluation

| Metric | Result |
|---|---|
| Extraction Accuracy | 90%+ on durable facts (career, projects, preferences) |
| Retrieval Precision (top-5) | 60-80% depending on query conceptual clarity |
| False Positives | Low; mostly hedged language correctly marked confidence 0.5 |
| Dedup Effectiveness | Multi-valued: ✅ no duplicates; Single-valued: ✅ updates in place |

Sample extraction quality: "I prefer Python and work on AI/ML projects" → 4 clean triples, no noise.

## Tech Stack

- **Backend:** FastAPI + SQLite + sentence-transformers (all-MiniLM-L6-v2)
- **Frontend:** Streamlit
- **LLM APIs:** Google Gemini (free tier)
- **Embeddings:** Local, no API calls
- **Deployment:** Streamlit Cloud (free)

## Deployment

### Local Demo
Already running! Just `streamlit run frontend/app.py`.

### Streamlit Cloud (Free)
```bash
git push  # your repo
# https://share.streamlit.io → Connect repo → Deploy
```

## Future Improvements

- Semantic tagging (auto-tag facts as "hobby", "career", "skill")
- LLM-based relevance scoring (replace embeddings for conceptual queries)
- Vector database backend (pgvector/Qdrant for production scale)
- Conflict detection + resolution
- Multi-user support + authentication
- Conversation branching / multi-thread memory

## License

MIT — use freely, fork, adapt.

---

Built as a portfolio project showcasing full-stack AI systems under real-world constraints (free APIs, local inference, minimal infra).