"""Run test questions through StudyMate and save results.

Usage:
    python evaluate.py                       # default settings
    python evaluate.py --top-k 6 --tag k6    # experiment with top-k
    python evaluate.py --tag chunk500        # after changing CHUNK_SIZE and re-running ingest.py

Automatic checks: refusal behaviour and whether the expected source was retrieved.
Relevance / faithfulness / correctness are filled in MANUALLY in the output CSV.
"""
import argparse
import csv
import time
from statistics import mean

from src.config import BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE, LLM_MODEL, TOP_K
from src.pipeline import ask
from src.prompts import NO_EVIDENCE_MESSAGE

EVAL_DIR = BASE_DIR / "eval"

FIELDS = [
    "question", "answerable", "expected_answer", "source_file",
    "generated_answer", "retrieved_sources", "best_distance",
    "refused", "refusal_correct", "source_hit",
    "retrieval_time_s", "generation_time_s", "total_time_s",
    "relevance", "faithfulness", "correctness", "notes",  # fill these by hand
]


def load_questions(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("question") or "").strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--tag", default="baseline", help="name added to the output file")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between questions")
    args = parser.parse_args()

    questions = load_questions(EVAL_DIR / "questions.csv")
    if not questions:
        print("No questions found in eval/questions.csv")
        return

    print(f"Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
          f"top_k={args.top_k}, llm={LLM_MODEL}, tag={args.tag}")

    # Warm-up so the first question does not include model loading time
    t0 = time.perf_counter()
    ask("warm up", k=args.top_k)
    print(f"Warm-up (cold start) took {time.perf_counter() - t0:.1f}s\n")

    rows = []
    for i, q in enumerate(questions, start=1):
        answerable = (q.get("answerable") or "").strip().lower() == "yes"
        out = ask(q["question"], k=args.top_k)

        refused = out["answer"] == NO_EVIDENCE_MESSAGE
        files = sorted({s["source"] for s in out["sources"]})
        best = min((s["distance"] for s in out["sources"]), default="")

        # Automatic checks
        refusal_correct = refused if not answerable else (not refused)
        expected_file = (q.get("source_file") or "").strip()
        source_hit = ""
        if answerable and expected_file:
            source_hit = expected_file in files

        rows.append({
            "question": q["question"],
            "answerable": "yes" if answerable else "no",
            "expected_answer": q.get("expected_answer", ""),
            "source_file": expected_file,
            "generated_answer": out["answer"],
            "retrieved_sources": "; ".join(
                f"{s['source']} p{s['page']} ({s['distance']})" for s in out["sources"]
            ),
            "best_distance": best,
            "refused": refused,
            "refusal_correct": refusal_correct,
            "source_hit": source_hit,
            "retrieval_time_s": round(out["retrieval_time"], 3),
            "generation_time_s": round(out["generation_time"], 3),
            "total_time_s": round(out["total_time"], 3),
            "relevance": "", "faithfulness": "", "correctness": "", "notes": "",
        })
        print(f"[{i}/{len(questions)}] {q['question'][:60]} -> "
              f"{'REFUSED' if refused else 'ANSWERED'} | total {out['total_time']:.2f}s")
        time.sleep(args.delay)  # avoid API rate limits

    out_path = EVAL_DIR / f"results_{args.tag}.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Summary (automatic checks only)
    ans = [r for r in rows if r["answerable"] == "yes"]
    unans = [r for r in rows if r["answerable"] == "no"]
    print("\n===== SUMMARY =====")
    print(f"Questions: {len(rows)} ({len(ans)} answerable, {len(unans)} unanswerable)")
    if ans:
        answered = sum(1 for r in ans if not r["refused"])
        hits = [r for r in ans if r["source_hit"] != ""]
        print(f"Answerable questions answered: {answered}/{len(ans)}")
        if hits:
            print(f"Expected source retrieved:     {sum(1 for r in hits if r['source_hit'])}/{len(hits)}")
    if unans:
        print(f"Unanswerable correctly refused: {sum(1 for r in unans if r['refused'])}/{len(unans)}")
    print(f"Avg retrieval time: {mean(r['retrieval_time_s'] for r in rows):.3f}s")
    llm_times = [r["generation_time_s"] for r in rows if r["generation_time_s"] > 0]
    if llm_times:
        print(f"Avg generation time: {mean(llm_times):.2f}s")
    print(f"\nSaved to {out_path}")
    print("Now open the CSV and fill relevance / faithfulness / correctness by hand.")


if __name__ == "__main__":
    main()