"""End-to-end RAG pipeline: question -> retrieval -> generation -> answer + sources."""
from typing import Any, Dict

from src.generator import generate_answer
from src.prompts import NO_EVIDENCE_MESSAGE
from src.retriever import retrieve


def ask(question: str) -> Dict[str, Any]:
    """Answer a student question from the lecture materials.

    Returns a dict with:
      answer, sources, retrieval_time, generation_time, total_time, used_llm
    Never raises for empty questions or LLM failures.
    """
    question = (question or "").strip()
    if not question:
        return {
            "answer": "Please type a question.",
            "sources": [],
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "total_time": 0.0,
            "used_llm": False,
        }

    try:
        results, r_time = retrieve(question)
    except Exception as e:
        return {
            "answer": f"[Error] Retrieval failed: {e}",
            "sources": [],
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "total_time": 0.0,
            "used_llm": False,
        }

    answer, g_time, used_llm = generate_answer(question, results)

    # Only show sources when the system actually answered from them
    sources = []
    if answer != NO_EVIDENCE_MESSAGE:
        for doc, dist in results:
            m = doc.metadata
            sources.append({
                "source": m.get("source", "?"),
                "subject": m.get("subject", "?"),
                "page": m.get("page", "?"),
                "distance": round(float(dist), 3),
                "text": doc.page_content,
            })

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_time": r_time,
        "generation_time": g_time,
        "total_time": r_time + g_time,
        "used_llm": used_llm,
    }


if __name__ == "__main__":
    q = input("Question: ")
    out = ask(q)
    print(f"\n{out['answer']}")
    print(f"\nSources ({len(out['sources'])}):")
    for s in out["sources"]:
        print(f"  - {s['source']} (page {s['page']}) distance={s['distance']}")
    print(
        f"\nRetrieval: {out['retrieval_time']:.3f}s | "
        f"Generation: {out['generation_time']:.2f}s | "
        f"Total: {out['total_time']:.2f}s | LLM called: {out['used_llm']}"
    )