import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
NOTES_PATH = os.path.join(BASE_DIR, "notes")

# Constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# LLM Provider ("ollama" or "gemini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama (LLM)
# NOTE: On CPU, large models can be extremely slow on prompt evaluation.
# This default is a small local model; we cap generation to avoid multi-minute "thinking" outputs.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "lfm2.5-thinking:latest")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "30"))

# Generation controls (lower values = faster)
# 512 is a practical cap that typically allows closing </think> + emitting the final answer.
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.05"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
OLLAMA_TOP_K = int(os.getenv("OLLAMA_TOP_K", "50"))

# Context trimming (reduces prompt_eval time on CPU)
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "1400"))
CONTEXT_MAX_CHUNK_CHARS = int(os.getenv("CONTEXT_MAX_CHUNK_CHARS", "350"))

COLLECTION_NAME = "inpt_notes"

# Retrieval settings  (tuned for speed)
VECTOR_K = int(os.getenv("VECTOR_K", "2"))
BM25_K = int(os.getenv("BM25_K", "2"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "2"))
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BM25_PAGE_SIZE = 5000
BM25_MAX_DOCS = 20000

# Query rewrite  (disabled — saves one full Ollama round-trip per query)
ENABLE_QUERY_REWRITE = False
REWRITE_MAX_WORDS = 10
