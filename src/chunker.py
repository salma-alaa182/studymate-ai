"""Split loaded pages into overlapping chunks, keeping metadata."""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into chunks. Metadata is copied to every chunk."""
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Give each chunk an id so we can cite it later
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


if __name__ == "__main__":
    from src.loader import load_all_documents

    docs = load_all_documents()
    chunks = chunk_documents(docs)
    print(f"Pages: {len(docs)} -> Chunks: {len(chunks)}")
    if chunks:
        lengths = [len(c.page_content) for c in chunks]
        print(f"Avg length: {sum(lengths) // len(lengths)} chars | Max: {max(lengths)}")
        print("\n--- Sample chunk ---")
        print("Metadata:", chunks[0].metadata)
        print(chunks[0].page_content)