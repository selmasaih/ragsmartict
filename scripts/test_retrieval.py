import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

def test_retrieval():
    print("Initialisation du client ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Erreur : Impossible de charger la collection. Avez-vous lancé l'ingestion ? ({e})")
        return

    print(f"Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    test_questions = [
        "Qu'est-ce que la transformée de Fourier ?",
        "Explique le théorème d'échantillonnage.",
        "Quelle est la différence entre un signal continu et un signal discret ?",
        "Qu'est-ce que la convolution de deux signaux ?",
        "Comment fonctionne un filtre passe-bas ?"
    ]

    for q in test_questions:
        print("-" * 50)
        print(f"Question : {q}")
        
        q_embedding = model.encode(q, normalize_embeddings=True).tolist()
        
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=3
        )
        
        if not results['documents'] or not results['documents'][0]:
            print("  Aucun résultat trouvé.")
            continue
            
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for i in range(len(docs)):
            print(f"\n  [Résultat {i+1}] Score (Distance): {distances[i]:.4f}")
            print(f"  Fichier: {metas[i]['filename']}, Page: {metas[i]['page_number']}")
            print(f"  Extrait: {docs[i][:150]}...")

if __name__ == "__main__":
    test_retrieval()
