"""
extraction.py
-------------
Turns a batch of conversation messages into structured (subject, predicate, object)
memory triples using Google's Gemini free-tier API.

Why Gemini free tier (Option B):
- 1500 requests/day free is plenty for dev + demo use.
- Materially better extraction consistency than a small local 7B model,
  which matters because the whole pitch of Memora is *clean* structured memory.

Setup:
1. Get a free key at https://aistudio.google.com/apikey
2. Put it in a .env file (see .env.example) as GEMINI_API_KEY=...
3. pip install google-generativeai python-dotenv
"""

import os
import json
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

_EXTRACTION_SYSTEM_PROMPT = """You extract long-term memory facts about a user from a conversation.

Return ONLY a JSON array (no markdown fences, no preamble) of objects with this exact shape:
[{"subject": "user", "predicate": "works_at", "object": "Acme Corp", "confidence": 0.95, "importance": 0.7}]

Rules:
- subject is almost always "user" unless the fact is clearly about someone/something else they mentioned.
- predicate MUST be chosen from this fixed list — do not invent new predicate names:
  is_a, academic_year, studies_at, studies_field, career_goal, works_on, participates_in,
  prefers, dislikes, enjoys, has_skill, owns, lives_in, has_goal, relationship
- If a predicate is naturally multi-valued (e.g. career_goal, works_on, participates_in, enjoys,
  has_skill, prefers), extract each value as its own separate triple with that same predicate —
  do not merge multiple values into one object, and do not treat a repeated predicate as a
  conflicting update.
- If a predicate is naturally single-valued (e.g. academic_year, studies_at, lives_in), a new
  value should be understood as replacing the old one.
- object is the value, kept concise (a few words, not a full sentence).
- confidence is 0-1: how certain you are this is a stated fact, not a guess.
- importance is 0-1: how useful this fact would be to remember for future personalization.
- Only extract durable facts — skip one-off situational statements.
- If there is nothing worth remembering, return an empty array: []
"""


import time
import random

def _get_client():
    import google.generativeai as genai
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file — see .env.example."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_EXTRACTION_SYSTEM_PROMPT,
    )


def _generate_with_backoff(model, prompt: str, max_retries: int = 4):
    """Retries on 429 (rate limit) with exponential backoff + jitter."""
    from google.api_core.exceptions import ResourceExhausted

    wait = 2
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except ResourceExhausted:
            if attempt == max_retries - 1:
                raise
            time.sleep(wait + random.uniform(0, 0.5))
            wait *= 2


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def extract_triples(messages: list[dict]) -> list[dict]:
    """
    messages: list of {"role": "user"/"assistant", "content": str}
    Returns: list of {"subject", "predicate", "object", "confidence", "importance"}
    """
    if not messages:
        return []

    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    model = _get_client()
    response = _generate_with_backoff(
        model, f"Conversation batch:\n\n{conversation_text}\n\nExtract memory facts as JSON array."
    )

    raw = _strip_code_fences(response.text)

    try:
        triples = json.loads(raw)
    except json.JSONDecodeError:
        # Model occasionally wraps output in explanatory text despite instructions.
        # Try to salvage the first [...] block.
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            triples = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    # Validate shape defensively — never trust model output blindly.
    valid = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        if not all(k in t for k in ("subject", "predicate", "object")):
            continue
        valid.append({
            "subject": str(t["subject"]).strip(),
            "predicate": str(t["predicate"]).strip().lower().replace(" ", "_"),
            "object": str(t["object"]).strip(),
            "confidence": float(t.get("confidence", 0.8)),
            "importance": float(t.get("importance", 0.5)),
        })
    return valid


if __name__ == "__main__":
    # Quick manual smoke test
    sample = [
        {"role": "user", "content": "I'm a second year AI/ML student, and I really prefer working in Python over Java."},
        {"role": "assistant", "content": "Got it, good to know!"},
        {"role": "user", "content": "I'm currently building a hackathon project called PULSE for SBI."},
    ]
    print(json.dumps(extract_triples(sample), indent=2))
