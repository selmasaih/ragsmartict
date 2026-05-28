# INPT Smart ICT Notes RAG

A smart virtual assistant based on the RAG (Retrieval-Augmented Generation) architecture to query your INPT Smart ICT course notes.

## ✨ Features

- 📡 **Streaming responses:** The answer is streamed token-by-token to the chat UI over Server-Sent Events (SSE), so you see it appear live instead of waiting for the full generation.
- 📄 **Optimized Ingestion:** Text extraction with `RecursiveCharacterTextSplitter` and batch embedding encoding for fast document ingestion.
- 🚀 **Parallel Hybrid Retrieval:** Simultaneous ChromaDB (vector) and BM25 (lexical) search via a `ThreadPoolExecutor`, followed by Cross-Encoder re-ranking.
- 🏷️ **Subject filtering:** Restrict retrieval to a single subject (folder) from the UI dropdown.
- 🧠 **Conversational Memory:** Follow-up questions keep the context of previous messages.
- 🗄️ **Vector Database:** Persistent local storage with ChromaDB.
- 🤖 **Pluggable LLM:** Local models via **Ollama** (full privacy) or cloud via the **Google Gemini API**, selected with one env var.
- 🔒 **Hardened API:** Configurable CORS allow-list, input-length validation, `/api/health` check, and structured logging.
- 🧪 **Tested:** Unit tests for the retrieval/generation helpers (including the streaming think-tag filter).

## 🏗️ Architecture

1. **Backend (FastAPI):** Async REST API exposing the RAG pipeline on port `8000`.
2. **Frontend (Vite):** Modern HTML/CSS/Vanilla-JS UI that consumes the streaming API.
3. **RAG Pipeline:**
   - **Ingestion:** Batch ingestion into ChromaDB (`src/ingest.py`).
   - **Retrieval:** Parallel BM25 + vector search → Cross-Encoder re-ranking (`src/query.py`).
   - **Generation:** LLM (Ollama/Gemini) augmented with conversation history, streamed back to the client.

### API endpoints

| Method | Path                 | Purpose                                            |
| ------ | -------------------- | -------------------------------------------------- |
| GET    | `/api/health`        | Liveness + collection status + active LLM provider |
| GET    | `/api/stats`         | Number of indexed chunks                           |
| GET    | `/api/topics`        | Distinct canonical topics available for filtering  |
| POST   | `/api/query`         | Non-streaming answer (JSON)                        |
| POST   | `/api/query/stream`  | Streaming answer (SSE: `sources` → `token`* → `done`) |

## 🚀 Installation & Usage

### Prerequisites
- Python 3.11+
- Node.js & npm (for the web interface)
- A local LLM provider (Ollama) or a cloud provider (Google Gemini API)

### 1. Backend configuration
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit it
```
Set `LLM_PROVIDER="ollama"` or `"gemini"` in `.env`. See `.env.example` for all options (CORS origins, model names, question size limit, log level, …).

### 2. Data ingestion
Place your PDFs under `notes/` (e.g. `notes/signal_processing/`). The sub-folder name becomes the document's **subject**.
```bash
python -m src.ingest            # add --reset to clear the collection first
```

### 3. Launch backend + frontend
On Windows, one script starts both:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
```
Or run them manually:
```bash
# Terminal 1 — API
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
# Terminal 2 — UI
cd frontend && npm install && npm run dev
```
Open the Vite link (usually `http://localhost:5173`).

> The frontend's API base URL can be overridden with the `VITE_API_BASE` env var (defaults to `http://127.0.0.1:8000/api`).

### 4. Run the tests
```bash
python -m pytest tests/ -q
```

*(The legacy Streamlit UI is still available via `streamlit run src/app.py`.)*
