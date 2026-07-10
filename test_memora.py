"""
test_memora.py
--------------
Systematic evaluation of Memora's extraction and retrieval quality.
Runs predefined test conversations, evaluates extraction output + retrieval ranking,
and produces a structured report for documentation.

Run with: python -m backend.test_memora
"""

import json
import requests
from datetime import datetime
from pathlib import Path

API_URL = "http://127.0.0.1:8000"
TEST_USER = "test_eval"
REPORT_PATH = Path(__file__).parent.parent / "EVALUATION_REPORT.json"

# Test conversations covering different fact types
TEST_CONVERSATIONS = [
    {
        "id": "test_career",
        "name": "Career Goals & Learning Style",
        "messages": [
            "I'm a second-year AI/ML student working toward careers in Generative AI and Conversational AI.",
            "I learn best by understanding the intuition behind things rather than memorizing.",
            "I prefer improving my own code instead of replacing it wholesale.",
        ],
        "queries": [
            {"q": "What are my career goals?", "expect": ["Generative AI", "Conversational AI"]},
            {"q": "How do I learn best?", "expect": ["understanding intuition", "improving code"]},
        ],
    },
    {
        "id": "test_hobbies",
        "name": "Hobbies & Interests",
        "messages": [
            "I enjoy solving DSA problems and I spend weekends watching Formula 1.",
            "I can solve a Rubik's Cube in under a minute.",
            "And I might also own a collection of vintage coins.",
        ],
        "queries": [
            {"q": "Do I have any hobbies?", "expect": ["Formula 1", "Rubik's Cube", "vintage coins"]},
            {"q": "What do I enjoy?", "expect": ["DSA", "Formula 1"]},
        ],
    },
    {
        "id": "test_projects",
        "name": "Current Projects",
        "messages": [
            "I'm building an AI memory layer called Memora.",
            "I'm also participating in the SBI Global FinTech Fest Hackathon 2026.",
        ],
        "queries": [
            {"q": "What am I currently working on?", "expect": ["Memora", "AI memory"]},
            {"q": "What hackathons am I in?", "expect": ["SBI", "FinTech"]},
        ],
    },
]

def run_conversation(conv_id: str, messages: list[str]) -> str:
    """Send messages to chat and return the conversation ID."""
    conversation_id = f"eval_{conv_id}_{int(datetime.now().timestamp())}"
    for msg in messages:
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={"user_id": TEST_USER, "conversation_id": conversation_id, "message": msg},
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"❌ Chat failed for '{msg}': {e}")
            return None
    return conversation_id

def trigger_extraction(conversation_id: str) -> dict:
    """Manually trigger extraction on a conversation."""
    try:
        resp = requests.post(
            f"{API_URL}/extract/{TEST_USER}/{conversation_id}",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return {"created_memory_ids": [], "count": 0}

def test_retrieval(query: str, expected_keywords: list[str]) -> dict:
    """Test a retrieval query and score how many expected keywords are found."""
    try:
        resp = requests.get(
            f"{API_URL}/debug/retrieve/{TEST_USER}",
            params={"query": query, "top_k": 5},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()
        
        # Score: how many expected keywords appear in the top-5 results
        found_keywords = set()
        for r in results:
            combined_text = f"{r['subject']} {r['predicate']} {r['object']}".lower()
            for keyword in expected_keywords:
                if keyword.lower() in combined_text:
                    found_keywords.add(keyword)
        
        score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
        
        return {
            "query": query,
            "expected_keywords": expected_keywords,
            "found_keywords": list(found_keywords),
            "retrieval_score": round(score, 2),
            "results": results,
        }
    except Exception as e:
        print(f"❌ Retrieval failed for '{query}': {e}")
        return {"query": query, "error": str(e)}

def main():
    print("\n🧪 Memora Evaluation Suite\n")
    print("=" * 60)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_conversations": [],
    }
    
    for conv in TEST_CONVERSATIONS:
        print(f"\n📝 Test: {conv['name']}")
        print("-" * 60)
        
        # Step 1: Run conversation
        print(f"  → Sending {len(conv['messages'])} messages...")
        conv_id = run_conversation(conv["id"], conv["messages"])
        if not conv_id:
            print(f"  ❌ Conversation failed")
            continue
        
        # Step 2: Trigger extraction
        print(f"  → Triggering extraction...")
        extract_result = trigger_extraction(conv_id)
        created_count = extract_result.get("count", 0)
        print(f"  ✅ Created {created_count} memories")
        
        # Step 3: Test retrieval for each query
        print(f"  → Testing {len(conv['queries'])} retrieval queries...")
        retrieval_results = []
        avg_score = 0
        for query_test in conv["queries"]:
            result = test_retrieval(query_test["q"], query_test["expect"])
            score = result.get("retrieval_score", 0)
            status = "✅" if score >= 0.6 else "⚠️" if score > 0 else "❌"
            print(f"    {status} '{query_test['q']}' → {score:.0%}")
            retrieval_results.append(result)
            avg_score += score
        
        avg_score /= len(conv["queries"]) if conv["queries"] else 0
        
        report["test_conversations"].append({
            "name": conv["name"],
            "messages_sent": len(conv["messages"]),
            "memories_created": created_count,
            "retrieval_tests": retrieval_results,
            "avg_retrieval_score": round(avg_score, 2),
        })
    
    # Write report
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Evaluation complete! Report saved to {REPORT_PATH}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()