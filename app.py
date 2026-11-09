"""StudyMate AI - Streamlit interface."""
import streamlit as st

from src.config import CHUNK_SIZE, LLM_MODEL, NO_EVIDENCE_DISTANCE, TOP_K
from src.pipeline import ask
from src.prompts import NO_EVIDENCE_MESSAGE

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")

# Keep question history across reruns
if "history" not in st.session_state:
    st.session_state.history = []


def render_result(out: dict) -> None:
    """Show answer, latency and sources for one result."""
    st.subheader("Answer")
    answer = out["answer"]
    if answer == NO_EVIDENCE_MESSAGE:
        st.info(answer)
    elif answer.startswith("[Error]"):
        st.error(answer)
    elif answer == "Please type a question.":
        st.warning(answer)
    else:
        st.write(answer)

    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieval", f"{out['retrieval_time']:.2f}s")
    c2.metric("Generation", f"{out['generation_time']:.2f}s")
    c3.metric("Total", f"{out['total_time']:.2f}s")

    if out["sources"]:
        st.subheader("Sources")
        for i, s in enumerate(out["sources"], start=1):
            label = f"[{i}] {s['source']} (page {s['page']}) | distance {s['distance']}"
            with st.expander(label):
                st.write(f"**Subject:** {s['subject']}")
                st.write(s["text"])


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "StudyMate AI uses Retrieval-Augmented Generation (RAG): it searches your "
        "lecture files, then answers using only what it finds."
    )
    st.divider()

    st.subheader("Settings")
    top_k = st.slider("Retrieved passages (top-k)", min_value=1, max_value=10, value=TOP_K)
    st.caption(
        f"Chunk size: {CHUNK_SIZE} | LLM: {LLM_MODEL} | "
        f"No-evidence threshold: {NO_EVIDENCE_DISTANCE}"
    )
    st.divider()

    st.subheader("Try asking")
    st.code("What is digital transformation?")

    if st.session_state.history:
        st.divider()
        st.subheader("History")
        for item in reversed(st.session_state.history[-8:]):
            st.write(f"• {item['question']}")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------- Main ----------
st.title("📚 StudyMate AI")
st.caption("Ask questions about your lecture materials. Answers are grounded in the documents.")

question = st.text_input("Your question", placeholder="e.g. What is digital transformation?")

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching the lectures and generating an answer..."):
            out = ask(question, k=top_k)
        st.session_state.history.append({"question": question.strip(), "out": out})
        render_result(out)