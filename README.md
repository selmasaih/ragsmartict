# INPT Smart ICT Notes RAG

Assistant virtuel basé sur une architecture RAG (Retrieval-Augmented Generation) pour interroger les notes de cours de la filière Smart ICT (INPT Rabat).

## ✨ Fonctionnalités

- 📡 **Réponses en streaming** — la réponse s'affiche token par token via Server-Sent Events (SSE).
- 🚀 **Recherche hybride parallèle** — ChromaDB (vectoriel) + BM25 (lexical) en parallèle, puis re-ranking par cross-encoder.
- 🏷️ **Classement automatique par domaine** — chaque document est rangé dans l'un des 8 domaines Smart ICT (nom + contenu), filtrable depuis l'UI.
- 📥 **Import de fichiers** — upload de PDF ou d'images ; extraction (PyMuPDF) + OCR (easyocr, sans Tesseract requis) + indexation à chaud.
- 🗂️ **Gestion des documents** — lister / filtrer / supprimer les documents indexés depuis l'interface.
- 🔢 **Citations numérotées** — extraits numérotés `[1] [2]…` cités dans la réponse et listés en dessous.
- 🧠 **Mémoire conversationnelle** — les questions de suivi gardent le contexte.
- 🤖 **LLM au choix** — local via **Ollama** (privé) ou cloud via **Google Gemini**.
- 🔒 **API durcie** — CORS configurable, validation des entrées, clé API optionnelle, rate limiting, `/api/health`, logs structurés.
- 🧪 **Testé + CI** — tests unitaires (helpers) et API (TestClient) ; GitHub Actions lance lint + tests + build frontend.
- 📊 **Harnais d'évaluation** — `scripts/eval_retrieval.py` mesure hit@k / MRR sur un jeu de questions labellisées.
- 🎨 **Spécialisation du style** — RAG + léger « style fine-tune » : Modelfile Ollama (sans GPU) et pipeline LoRA optionnel (`finetune/`).

## 🏗️ Architecture

1. **Backend (FastAPI)** — API REST asynchrone exposant le pipeline RAG (port `8000`).
2. **Frontend (Vite)** — interface HTML/CSS/JS qui consomme l'API en streaming.
3. **Pipeline RAG**
   - **Ingestion** (`src/ingest.py`) — extraction PyMuPDF/OCR → chunking → classement par domaine → embeddings e5 (`passage:`) → ChromaDB.
   - **Recherche** (`src/query.py`) — BM25 + vectoriel en parallèle → re-ranking cross-encoder → diversification (cap par document).
   - **Génération** — LLM (Ollama/Gemini) augmenté de l'historique, streamé au client.

### Endpoints

| Méthode | Chemin                | Rôle                                                  |
| ------- | --------------------- | ----------------------------------------------------- |
| GET     | `/api/health`         | État + nombre de chunks + provider LLM                |
| GET     | `/api/stats`          | Nombre de chunks indexés                              |
| GET     | `/api/topics`         | Domaines disponibles pour le filtre                   |
| GET     | `/api/documents`      | Liste des documents (nom, domaine, nb d'extraits)     |
| DELETE  | `/api/documents`      | Supprime un document *(clé API si configurée)*        |
| POST    | `/api/upload`         | Importe et indexe un PDF/image *(clé API si configurée)* |
| POST    | `/api/query`          | Réponse non-streamée (JSON)                           |
| POST    | `/api/query/stream`   | Réponse streamée (SSE : `sources` → `token`* → `done`) |

## 🚀 Installation & utilisation

### Prérequis
- Python 3.11+
- Node.js & npm (pour l'interface)
- Ollama (local) ou une clé Google Gemini (cloud)

### 1. Backend
```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # puis éditer
```
Choisir `LLM_PROVIDER="ollama"` ou `"gemini"`. Voir `.env.example` pour toutes les options (CORS, modèles, `API_KEY`, `RATE_LIMIT`, `BM25_MAX_DOCS`, …).

### 2. Ingestion
Placer les fichiers (PDF/images) sous `notes/` (le nom du sous-dossier sert d'origine ; le domaine est déduit automatiquement).
```bash
python -m src.ingest            # --reset pour repartir de zéro
```
Reclasser les documents existants : `python -m scripts.classify_topics` (édite `topic_mapping.tsv`) puis `python -m scripts.apply_topics`.

### 3. Lancer backend + frontend
```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
```
Ou manuellement :
```bash
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000   # API
cd frontend && npm install && npm run dev                              # UI
```
> URL de l'API côté frontend surchargée par `VITE_API_BASE` (défaut `http://127.0.0.1:8000/api`).

### 4. Tests & évaluation
```bash
python -m pytest tests/ -q          # tests unitaires + API
python -m scripts.eval_retrieval    # hit@k / MRR du retrieval
```

### 5. Docker (backend)
```bash
docker build -t inpt-rag .
docker run -p 8000:8000 -v "$PWD/chroma_db:/app/chroma_db" -v "$PWD/notes:/app/notes" inpt-rag
```

## 🔐 Sécurité
- `CORS_ORIGINS` — liste blanche d'origines (ne pas laisser `*` en prod).
- `API_KEY` — si définie, header `X-API-Key` requis sur `upload` / `query` / `delete`.
- `RATE_LIMIT` — limite par client (défaut `30/minute`).
- La sortie markdown du LLM est nettoyée (DOMPurify) côté client.
