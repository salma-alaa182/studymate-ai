"""Embedding model used for both chunks and questions."""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and reuse it."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # unit vectors -> cosine similarity
    )