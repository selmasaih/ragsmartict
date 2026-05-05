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

# ── LLM provider ("ollama" or "gemini") ──────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# ── Ollama settings ──────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_TIMEOUT_S = 300
OLLAMA_NUM_PREDICT = 200          # max tokens — shorter = faster
OLLAMA_TEMPERATURE = 0.3
OLLAMA_TOP_P = 0.9
OLLAMA_TOP_K = 40

# ── Embedding / Chunking ─────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "inpt_notes"

# ── Retrieval settings ───────────────────────────────────────────────
TOP_K = 5
VECTOR_K = 10                     # was 15 — fewer vector candidates
BM25_K = 10                       # was 25 — fewer BM25 candidates
RERANK_TOP_K = 5                  # was 6
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BM25_PAGE_SIZE = 5000
BM25_MAX_DOCS = 20000             # was 50000 — halved

# ── Context window limits ────────────────────────────────────────────
CONTEXT_MAX_CHARS = 4000          # total context sent to LLM
CONTEXT_MAX_CHUNK_CHARS = 800     # per-chunk cap

# ── Query rewrite (DISABLED — saves a full LLM round-trip) ──────────
ENABLE_QUERY_REWRITE = False
REWRITE_MAX_WORDS = 10
