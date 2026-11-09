"""Retrieve the most relevant chunks for a student question."""
import time
from typing import List, Tuple

from langchain_core.documents import Document

from src.config import TOP_K
from src.vectorstore import load_vectorstore

_vectorstore = None


def _get_store():
    """Open Chroma once and reuse it for every question."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def retrieve(question: str, k: int = TOP_K) -> Tuple[List[Tuple[Document, float]], float]:
    """Return ([(chunk, distance), ...], retrieval_seconds).

    Lower distance = more similar (cosine distance).
    An empty question returns no results instead of crashing.
    """
    question = (question or "").strip()
    if not question:
        return [], 0.0

    t0 = time.perf_counter()
    results = _get_store().similarity_search_with_score(question, k=k)
    elapsed = time.perf_counter() - t0
    return results, elapsed


if __name__ == "__main__":
    q = input("Question: ")
    results, secs = retrieve(q)
    print(f"\nRetrieved {len(results)} chunks in {secs:.3f}s")
    for doc, dist in results:
        m = doc.metadata
        print(f"\n[distance={dist:.3f}] {m['source']} (page {m['page']})")
        print(doc.page_content[:200])