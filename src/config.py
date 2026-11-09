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

# LLM (Gemini). Check the current model name in Google AI Studio
LLM_MODEL = "gemini-3.8-flash"
LLM_TEMPERATURE = 0.0

# Chroma
COLLECTION_NAME = "studymate"

NO_EVIDENCE_DISTANCE = 0.65