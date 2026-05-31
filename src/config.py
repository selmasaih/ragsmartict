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

# ── Uploads (multi-tenant safety) ────────────────────────────────────
# Uploaded docs are tagged with this subject, kept OUT of the shared global
# search, and only retrievable by their owner's session. They expire after
# UPLOAD_TTL_HOURS to avoid unbounded growth of the shared DB.
UPLOAD_SUBJECT = "Uploads"
UPLOAD_TTL_HOURS = _env_int("UPLOAD_TTL_HOURS", 24)
# Optional API key: when set, mutating endpoints require header X-API-Key.
API_KEY = os.getenv("API_KEY") or None
# Per-client rate limit for query/upload endpoints (slowapi syntax).
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

# ── LLM provider ("ollama", "gemini" or "groq") ──────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# ── Groq settings (free, fast cloud inference — recommended for deploy) ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or None
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Smaller model used automatically when the primary hits a rate limit (429)
# or errors — keeps the app responsive instead of failing outright.
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Cloud models are fast — allow long, thorough answers (local Ollama uses
# OLLAMA_NUM_PREDICT instead, kept small for CPU speed).
GROQ_MAX_TOKENS = _env_int("GROQ_MAX_TOKENS", 1024)

# ── Ollama settings ──────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
# Base URL of the Ollama server; specific endpoints are derived from it.
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_KEEP_ALIVE = "10m"
OLLAMA_TIMEOUT_S = _env_int("OLLAMA_TIMEOUT_S", 300)
# Cap output length for speed on CPU (fewer tokens = faster total latency).
OLLAMA_NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", 220)
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
TOP_K = 6
# Candidate pool for the cross-encoder rerank. Larger = richer material for
# the answer (slightly slower CPU rerank).
VECTOR_K = _env_int("VECTOR_K", 12)
BM25_K = _env_int("BM25_K", 8)
# Set ENABLE_RERANK=false on memory-constrained hosts (the cross-encoder adds
# ~400 MB RAM + CPU latency); retrieval still works via vector+BM25 scores.
ENABLE_RERANK = _env_bool("ENABLE_RERANK", True)
RERANK_TOP_K = 6
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BM25_PAGE_SIZE = 5000
# Max chunks indexed for lexical (BM25) search. 0 = no cap (index everything).
BM25_MAX_DOCS = _env_int("BM25_MAX_DOCS", 0)

# ── Context window limits ────────────────────────────────────────────
# More context = richer, more thorough answers (cloud LLMs handle it fast).
CONTEXT_MAX_CHARS = _env_int("CONTEXT_MAX_CHARS", 5000)   # total context to LLM
CONTEXT_MAX_CHUNK_CHARS = 900     # per-chunk cap
MAX_CHUNKS_PER_DOC = _env_int("MAX_CHUNKS_PER_DOC", 3)  # diversify sources

# ── Query rewrite (DISABLED — saves a full LLM round-trip) ──────────
ENABLE_QUERY_REWRITE = _env_bool("ENABLE_QUERY_REWRITE", False)
REWRITE_MAX_WORDS = 10

# ── Style specialization (lightweight "style fine-tune" without training) ──
# Injects a short worked example into the system prompt to anchor the answer
# style (structure, scientific tone, [n] citations). Pairs with the Ollama
# Modelfile in finetune/.
STYLE_FEWSHOT = _env_bool("STYLE_FEWSHOT", True)


def validate() -> list[str]:
    """Return a list of human-readable configuration problems (empty if OK)."""
    problems = []
    if LLM_PROVIDER not in {"ollama", "gemini", "groq"}:
        problems.append(f"LLM_PROVIDER must be 'ollama', 'gemini' or 'groq', got '{LLM_PROVIDER}'.")
    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        problems.append("LLM_PROVIDER is 'gemini' but GOOGLE_API_KEY is not set.")
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        problems.append("LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set.")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    return problems
