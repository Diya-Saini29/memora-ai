"""
debug_extraction.py
--------------------
One-off script to see Gemini's raw response before JSON parsing swallows errors.
Run with: python -m backend.debug_extraction
"""
from backend import extraction

sample = [
    {"role": "user", "content": "I'm a second year AI/ML student at Thapar and I prefer Python over Java."},
    {"role": "assistant", "content": "Good to know!"},
    {"role": "user", "content": "I'm currently building a project called Memora, a personal AI memory layer."},
]

conversation_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in sample)
model = extraction._get_client()
response = model.generate_content(
    f"Conversation batch:\n\n{conversation_text}\n\nExtract memory facts as JSON array."
)
print("RAW RESPONSE:")
print(repr(response.text))
print()
print("PARSED:")
print(extraction.extract_triples(sample))