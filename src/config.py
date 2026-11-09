"""Central configuration for StudyMate AI."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Embeddings (free, runs locally)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval
TOP_K = 4

# LLM (Gemini)
LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TEMPERATURE = 0.0

# If the closest chunk is farther than this (cosine distance), skip the LLM
# and answer "not enough information". Tune it using your tests.
NO_EVIDENCE_DISTANCE = 0.65

# Chroma
COLLECTION_NAME = "studymate"