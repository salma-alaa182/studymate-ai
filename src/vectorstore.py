"""Build and load the Chroma vector database."""
import shutil
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_DIR, COLLECTION_NAME
from src.embeddings import get_embeddings


def build_vectorstore(chunks: List[Document], reset: bool = True) -> Chroma:
    """Embed chunks and store them in Chroma. reset=True rebuilds from scratch."""
    if not chunks:
        raise ValueError("No chunks to index. Check that data/ contains readable files.")

    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vectorstore() -> Chroma:
    """Open the existing Chroma database."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("chroma_db not found. Run `python ingest.py` first.")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )