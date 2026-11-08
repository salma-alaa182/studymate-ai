"""Run once to build the vector database from data/."""
import time

from src.loader import load_all_documents
from src.chunker import chunk_documents
from src.vectorstore import build_vectorstore


def main():
    t0 = time.perf_counter()
    docs = load_all_documents()
    print(f"Loaded {len(docs)} pages")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    t1 = time.perf_counter()
    build_vectorstore(chunks)
    t2 = time.perf_counter()

    print(f"Embedding + indexing time: {t2 - t1:.1f}s")
    print(f"Total time: {t2 - t0:.1f}s")
    print("Index saved to chroma_db/")


if __name__ == "__main__":
    main()