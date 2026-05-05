# INPT Smart ICT Notes RAG

Un assistant intelligent basé sur l'architecture RAG pour interroger vos notes de cours de la filière Smart ICT à l'INPT.
*A smart RAG-based assistant to query your INPT Smart ICT course notes.*

## 📸 Démo
*(Placeholder for screenshot)*
![Capture d'écran de l'application](docs/screenshots/app_demo.png)

🔗 **Lien de la démo en direct :** [Live URL Placeholder]

## ✨ Fonctionnalités
- 📄 **Ingestion de PDF :** Extraction de texte à partir de documents PDF locaux.
- ✂️ **Découpage intelligent :** Utilisation de `RecursiveCharacterTextSplitter` pour découper le texte de manière sémantique.
- 🧠 **Embeddings Multilingues :** Encodage avec le modèle `intfloat/multilingual-e5-small`.
- 🗄️ **Base de Données Vectorielle :** Stockage local persistant avec ChromaDB.
- 🤖 **Génération de Réponses :** Supporte à la fois un modèle local via **Ollama** (ex: `lfm2.5-thinking`) pour la confidentialité totale, et l'API **Google Gemini** (`gemini-1.5-flash`) pour des performances rapides.
- 🏷️ **Citations :** Chaque réponse est accompagnée de la source (nom du fichier et page).
- 💻 **Interface Utilisateur :** Interface web interactive développée avec Streamlit.

## 🏗️ Architecture
L'architecture de l'application repose sur un pipeline RAG classique :
1. **Préparation des données :** Extraction, découpage et création d'embeddings à partir des PDF.
2. **Stockage :** Sauvegarde des vecteurs dans ChromaDB.
3. **Interrogation :** Création d'un embedding pour la question de l'utilisateur.
4. **Récupération (Retrieval) :** Recherche des chunks de texte les plus pertinents.
5. **Génération :** Envoi du contexte récupéré et de la question au modèle Gemini pour formuler la réponse.

Pour plus de détails, consultez [Architecture](docs/architecture.md).

## 🚀 Installation et Utilisation

### Prérequis
- Python 3.11
- Un fournisseur LLM local (Ollama) ou cloud (Google Gemini API)

### Étapes d'installation
1. Clonez ce dépôt.
2. Créez un environnement virtuel et installez les dépendances :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Créez un fichier `.env` basé sur `.env.example` et configurez votre fournisseur (`LLM_PROVIDER="ollama"` ou `"gemini"`). Si vous utilisez Gemini, ajoutez votre `GOOGLE_API_KEY`.
4. Placez vos PDF dans le dossier `notes/` (ex: `notes/signal_processing/`).

### Ingestion des données
Exécutez le script d'ingestion pour traiter les PDF et remplir la base de données :
```bash
python src/ingest.py
```
*(Utilisez l'option `--reset` pour vider la base avant l'ingestion).*

### Lancement de l'interface Streamlit
```bash
streamlit run src/app.py
```

### Tester la récupération (Retrieval)
```bash
python scripts/test_retrieval.py
```
