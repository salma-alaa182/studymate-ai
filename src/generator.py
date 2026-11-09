"""Call Gemini with the grounded prompt and return the answer."""
import os
import time
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL, LLM_TEMPERATURE, NO_EVIDENCE_DISTANCE
from src.prompts import NO_EVIDENCE_MESSAGE, build_prompt

load_dotenv()

_llm = None


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini client once."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your .env file.")
        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            google_api_key=api_key,
            max_retries=3,
            timeout=60,
        )
    return _llm


def _to_text(content) -> str:
    """Gemini may return a string or a list of parts. Always give back text."""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts).strip()


def generate_answer(
    question: str, results: List[Tuple[Document, float]]
) -> Tuple[str, float, bool]:
    """Return (answer, generation_seconds, used_llm).

    used_llm is False when we short-circuited with the no-evidence message.
    """
    if not results:
        return NO_EVIDENCE_MESSAGE, 0.0, False

    best_distance = min(dist for _, dist in results)
    if best_distance > NO_EVIDENCE_DISTANCE:
        return NO_EVIDENCE_MESSAGE, 0.0, False

    system, user = build_prompt(question, results)
    t0 = time.perf_counter()
    try:
        response = _get_llm().invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        answer = _to_text(response.content)
    except Exception as e:
        return f"[Error] Could not get an answer from the LLM: {e}", 0.0, True
    elapsed = time.perf_counter() - t0

    return (answer or NO_EVIDENCE_MESSAGE), elapsed, True


if __name__ == "__main__":
    from src.retriever import retrieve

    q = input("Question: ")
    results, r_time = retrieve(q)
    if results:
        print(f"Closest distance: {min(d for _, d in results):.3f}")
    answer, g_time, used = generate_answer(q, results)
    print(f"\n{answer}")
    print(f"\nRetrieval: {r_time:.3f}s | Generation: {g_time:.2f}s | LLM called: {used}")