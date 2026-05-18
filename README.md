# INPT Smart ICT Notes RAG

A smart virtual assistant based on the RAG (Retrieval-Augmented Generation) architecture to query your INPT Smart ICT course notes.

## ✨ Features and Recent Improvements

- 📄 **Optimized Ingestion:** Text extraction using `RecursiveCharacterTextSplitter`. Employs batch encoding for embeddings to massively reduce document ingestion time.
- 🚀 **Parallel Hybrid Retrieval:** Simultaneous search in ChromaDB (Vector) and BM25 (Lexical) using a `ThreadPoolExecutor` to minimize latency.
- 🧠 **Conversational Memory:** The assistant remembers the context of previous messages, allowing you to ask follow-up questions naturally.
- 💎 **Premium User Interface:** A brand new web interface (Vite) featuring a stunning dark glassmorphism design, modern typography, and smooth animations.
- 🗄️ **Vector Database:** Persistent local storage with ChromaDB.
- 🤖 **Response Generation:** Supports local models via **Ollama** (`llama3.2:3b` or similar) for total privacy, or cloud generation via the **Google Gemini API**.
- 🏷️ **Citations:** Every response is accompanied by its source (document name and page number).

## 🏗️ Architecture

1. **Backend (FastAPI):** Asynchronous REST API managing the RAG pipeline, exposed on port `8000`.
2. **Frontend (Vite):** Modern user interface (HTML/CSS/Vanilla JS) connected to the backend.
3. **RAG Pipeline:**
   - **Ingestion:** Batch ingestion into ChromaDB (`src/ingest.py`).
   - **Retrieval:** Parallel BM25 + Vector Search, followed by Cross-Encoder Re-Ranking (`src/query.py`).
   - **Generation:** LLM augmented with conversation history (Memory).

## 🚀 Installation & Usage

### Prerequisites
- Python 3.11+
- Node.js & npm (for the graphical interface)
- A local LLM provider (Ollama) or a cloud provider (Google Gemini API)

### 1. Backend Configuration
```bash
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure the environment
cp .env.example .env
```
Edit the `.env` file according to your preferences (`LLM_PROVIDER="ollama"` or `"gemini"`).

### 2. Data Ingestion
Place your PDF files in the `notes/` directory (e.g., `notes/signal_processing/`).
Run the ingestion script to encode your documents into the database:
```bash
python src/ingest.py
```
*(Use the `--reset` flag to clear the database before ingestion if needed).*

### 3. Launch the Backend API (FastAPI)
Start the API that powers the RAG intelligence:
```bash
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Launch the Frontend Interface (Vite)
Open a new terminal and start the web application:
```bash
cd frontend
npm install
npm run dev
```
Then, open the local link provided by Vite (usually `http://localhost:5173`) in your web browser to enjoy the experience!

*(Note: The old Streamlit interface via `streamlit run src/app.py` is still available in the codebase if you prefer a basic UI).*
