"""Prompt template that grounds the LLM in the retrieved context."""
from typing import List, Tuple

from langchain_core.documents import Document

# The exact sentence the model must use when the context is not enough.
# We also use it in generator.py to detect and short-circuit "no evidence" cases.
NO_EVIDENCE_MESSAGE = (
    "I don't have enough information in the lecture materials to answer this question."
)

SYSTEM_PROMPT = f"""You are StudyMate AI, a study assistant for university students.

Rules:
1. Answer ONLY using the numbered context passages below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, reply exactly with:
   "{NO_EVIDENCE_MESSAGE}"
3. Do not guess, invent facts, or add details that are not in the context.
4. Be clear and concise. Use simple language suited for students.
5. After the answer, cite the passages you used like this: (Sources: [1], [3]).
"""

USER_TEMPLATE = """Context passages:
{context}

Student question: {question}

Answer:"""


def format_context(results: List[Tuple[Document, float]]) -> str:
    """Turn retrieved chunks into a numbered block with source info."""
    blocks = []
    for i, (doc, _dist) in enumerate(results, start=1):
        m = doc.metadata
        header = f"[{i}] {m.get('source', '?')} (page {m.get('page', '?')})"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_prompt(question: str, results: List[Tuple[Document, float]]) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) ready to send to the LLM."""
    user = USER_TEMPLATE.format(context=format_context(results), question=question)
    return SYSTEM_PROMPT, user


if __name__ == "__main__":
    from src.retriever import retrieve

    q = input("Question: ")
    results, _ = retrieve(q)
    system, user = build_prompt(q, results)
    print("=== SYSTEM ===")
    print(system)
    print("=== USER ===")
    print(user)