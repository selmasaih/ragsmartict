# Architecture du projet INPT Smart ICT Notes RAG

L'application suit une architecture RAG (Retrieval-Augmented Generation) divisée en deux phases principales : l'ingestion et l'interrogation.

## Phase d'Ingestion (`src/ingest.py`)
1. **Lecture des documents :** Le script parcourt le dossier `notes/` pour trouver les fichiers PDF et utilise `pypdf` pour extraire le texte page par page.
2. **Nettoyage et Découpage :** Les pages de moins de 50 caractères sont ignorées. Le texte est découpé avec `RecursiveCharacterTextSplitter` (800 caractères par chunk, 100 de chevauchement).
3. **Génération d'Embeddings :** Les chunks sont convertis en vecteurs avec le modèle `sentence-transformers` (`intfloat/multilingual-e5-small`).
4. **Stockage :** Les vecteurs, ainsi que les métadonnées (nom du fichier, sujet, page), sont stockés dans une collection ChromaDB.

## Phase d'Interrogation (`src/query.py`)
1. **Embedding de la Question :** La question de l'utilisateur est convertie en vecteur via le même modèle d'embedding.
2. **Recherche de similarité :** ChromaDB renvoie les `TOP_K` (5) chunks les plus pertinents.
3. **Génération de Prompt :** Un prompt système en français définit le rôle de l'assistant. Le contexte récupéré et la question sont combinés.
4. **Appel LLM :** L'API `gemini-1.5-flash` génère une réponse basée **uniquement** sur le contexte fourni, en citant ses sources.

## Interface (`src/app.py`)
L'interface Streamlit permet à l'utilisateur d'interagir facilement avec le système, d'afficher la réponse et d'inspecter les extraits sources retournés par la recherche vectorielle.
