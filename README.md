# INPT Smart ICT Notes RAG

Un assistant intelligent basé sur l'architecture RAG pour interroger vos notes de cours de la filière Smart ICT à l'INPT.
*A smart RAG-based assistant to query your INPT Smart ICT course notes.*

## ✨ Fonctionnalités et Améliorations Récentes

- 📄 **Ingestion Optimisée :** Extraction de texte avec `RecursiveCharacterTextSplitter`. Traitement par lots (Batch Encoding) des embeddings pour réduire massivement le temps d'ingestion.
- 🚀 **Retrieval Parallèle Hybride :** Recherche simultanée dans ChromaDB (Vectoriel) et via BM25 (Lexical) avec un `ThreadPoolExecutor` pour minimiser la latence.
- 🧠 **Mémoire Conversationnelle :** L'assistant mémorise le contexte des messages précédents pour vous permettre de poser des questions de suivi naturellement.
- 💎 **Interface Utilisateur Premium :** Nouvelle interface web (Vite) avec un design dark glassmorphism très esthétique, typographie moderne et animations fluides.
- 🗄️ **Base de Données Vectorielle :** Stockage local persistant avec ChromaDB.
- 🤖 **Génération de Réponses :** Supporte le local via **Ollama** (`llama3.2:3b` ou autre) ou le cloud via l'API **Google Gemini**.
- 🏷️ **Citations :** Chaque réponse est accompagnée de sa source (nom du document et page).

## 🏗️ Architecture

1. **Backend (FastAPI) :** API REST asynchrone gérant le RAG, exposée sur le port `8000`.
2. **Frontend (Vite) :** Interface utilisateur moderne (HTML/CSS/Vanilla JS) connectée au backend.
3. **Pipeline RAG :**
   - **Ingestion :** Ingestion batch dans ChromaDB (`src/ingest.py`).
   - **Récupération (Retrieval) :** Recherche parallèle BM25 + Vectorielle, puis Cross-Encoder Re-Ranking (`src/query.py`).
   - **Génération :** LLM augmenté de l'historique de conversation (Mémoire).

## 🚀 Installation et Utilisation

### Prérequis
- Python 3.11+
- Node.js & npm (pour l'interface graphique)
- Un fournisseur LLM local (Ollama) ou cloud (Google Gemini API)

### 1. Configuration du Backend
```bash
# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer l'environnement
cp .env.example .env
```
Éditez le fichier `.env` selon vos préférences (`LLM_PROVIDER="ollama"` ou `"gemini"`).

### 2. Ingestion des données
Placez vos PDF dans le dossier `notes/` (ex: `notes/signal_processing/`).
Exécutez le script d'ingestion pour encoder vos documents dans la base :
```bash
python src/ingest.py
```
*(Utilisez l'option `--reset` pour vider la base avant l'ingestion).*

### 3. Lancer l'API Backend (FastAPI)
L'API qui gère l'intelligence du système RAG doit être démarrée :
```bash
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Lancer l'Interface Frontend (Vite)
Ouvrez un nouveau terminal et lancez l'application web :
```bash
cd frontend
npm install
npm run dev
```
Ouvrez ensuite le lien local fourni par Vite (généralement `http://localhost:5173`) dans votre navigateur pour profiter de l'expérience !

*(Note : L'ancienne interface Streamlit via `streamlit run src/app.py` reste disponible dans le code si vous préférez une UI basique)*
