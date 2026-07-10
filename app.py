"""
app.py
------
Streamlit UI for Memora with custom dark mode + pastel aesthetic.
Dark tech-forward base with soft, soothing pastel accents.
"""

import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

# Custom CSS for dark mode + pastel aesthetic

CUSTOM_CSS = """
<style>
:root {
  --primary: #d8c8e8;
  --secondary: #b8e0e0;
}

h1, h2, h3 {
  color: var(--primary);
}

hr {
  border-color: var(--primary);
  opacity: 0.3;
}

a {
  color: var(--secondary);
}
</style>
"""
st.set_page_config(
    page_title="Memora",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.markdown("# 🧠")
with col2:
    st.markdown("# Memora")
st.caption("A personalized AI memory layer — structured knowledge graphs of what it learns about you.")

tab_chat, tab_memories = st.tabs(["💬 Chat", "🗂️ Memory Dashboard"])

with tab_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("retrieved_memories"):
                with st.expander("💾 Memories used"):
                    for m in msg["retrieved_memories"]:
                        st.markdown(
                            f"**{m['subject']}** → *{m['predicate'].replace('_', ' ')}* → **{m['object']}**"
                        )

    user_input = st.chat_input("Say something...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("✨ Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "user_id": st.session_state.user_id,
                        "conversation_id": st.session_state.conversation_id,
                        "message": user_input,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("❌ Can't reach the backend. Is `uvicorn backend.api:app` running?")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"Backend error: {e}")
                st.stop()

        with st.chat_message("assistant"):
            st.write(data["reply"])
            if data["retrieved_memories"]:
                with st.expander("💾 Memories used"):
                    for m in data["retrieved_memories"]:
                        st.markdown(
                            f"**{m['subject']}** → *{m['predicate'].replace('_', ' ')}* → **{m['object']}** (score: {m['score']:.2f})"
                        )
            if data["extraction_triggered"]:
                st.success("🧩 New memories extracted from this conversation.")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": data["reply"],
            "retrieved_memories": data["retrieved_memories"],
        })

with tab_memories:
    st.subheader("Stored Memories")
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/memories/{st.session_state.user_id}", timeout=10)
        resp.raise_for_status()
        memories = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Can't reach the backend. Is `uvicorn backend.api:app` running?")
        memories = []

    if not memories:
        st.info("📝 No memories yet — chat for a while or use the manual extract button below.")
    else:
        for m in memories:
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"**{m['subject']}** → *{m['predicate'].replace('_', ' ')}* → **{m['object']}**")
                st.caption(
                    f"confidence: {m['confidence']:.2f} · importance: {m['importance_score']:.2f} · {m['created_at']}"
                )
            with col2:
                if st.button("🗑️", key=f"del_{m['id']}", use_container_width=True):
                    requests.delete(f"{API_URL}/memories/{m['id']}")
                    st.rerun()
            st.divider()

    st.subheader("Manual Extraction")
    st.caption("Force extraction on the current conversation without waiting for the batch threshold.")
    if st.button("🧪 Run extraction now", use_container_width=True):
        try:
            resp = requests.post(
                f"{API_URL}/extract/{st.session_state.user_id}/{st.session_state.conversation_id}",
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            st.success(f"✅ Created {result['count']} new memories.")
        except Exception as e:
            st.error(f"❌ Extraction failed: {e}")