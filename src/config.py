import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# Environment Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
NOTES_PATH = os.path.join(BASE_DIR, "notes")

# ── Server / API ─────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Comma-separated list of allowed CORS origins. Use "*" to allow all (dev only).
CORS_ORIGINS = _env_list("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
MAX_QUESTION_CHARS = _env_int("MAX_QUESTION_CHARS", 2000)
MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 6)
# Optional API key: when set, mutating endpoints require header X-API-Key.
API_KEY = os.getenv("API_KEY") or None
# Per-client rate limit for query/upload endpoints (slowapi syntax).
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

# ── LLM provider ("ollama" or "gemini") ──────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # "ollama" or "gemini"

# ── Ollama settings ──────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
# Base URL of the Ollama server; specific endpoints are derived from it.
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_TIMEOUT_S = _env_int("OLLAMA_TIMEOUT_S", 300)
OLLAMA_NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", 512)
OLLAMA_TEMPERATURE = 0.3
OLLAMA_TOP_P = 0.9
OLLAMA_TOP_K = 40

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Embedding / Chunking ─────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "inpt_notes"

# ── Retrieval settings ───────────────────────────────────────────────
TOP_K = 5
VECTOR_K = 15
BM25_K = 10
ENABLE_RERANK = True
RERANK_TOP_K = 5
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BM25_PAGE_SIZE = 5000
BM25_MAX_DOCS = 20000

# ── Context window limits ────────────────────────────────────────────
CONTEXT_MAX_CHARS = 4000          # total context sent to LLM
CONTEXT_MAX_CHUNK_CHARS = 800     # per-chunk cap
MAX_CHUNKS_PER_DOC = _env_int("MAX_CHUNKS_PER_DOC", 2)  # diversify sources

# ── Query rewrite (DISABLED — saves a full LLM round-trip) ──────────
ENABLE_QUERY_REWRITE = _env_bool("ENABLE_QUERY_REWRITE", False)
REWRITE_MAX_WORDS = 10


def validate() -> list[str]:
    """Return a list of human-readable configuration problems (empty if OK)."""
    problems = []
    if LLM_PROVIDER not in {"ollama", "gemini"}:
        problems.append(f"LLM_PROVIDER must be 'ollama' or 'gemini', got '{LLM_PROVIDER}'.")
    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        problems.append("LLM_PROVIDER is 'gemini' but GOOGLE_API_KEY is not set.")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    return problems
