import streamlit as st
import chromadb
from src.query import answer_question
from src.config import CHROMA_DB_PATH, COLLECTION_NAME, GOOGLE_API_KEY

st.set_page_config(
    page_title="INPT Smart ICT Notes RAG",
    page_icon="📚",
    layout="wide"
)

# Sidebar
st.sidebar.title("📚 INPT Smart ICT Notes RAG")
st.sidebar.markdown(
    "Un assistant intelligent basé sur vos notes de cours (filière Smart ICT). "
    "Posez vos questions et obtenez des réponses sourcées !"
)

def get_doc_count():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0

doc_count = get_doc_count()
st.sidebar.metric(label="📄 Documents indexés (Chunks)", value=doc_count)

st.sidebar.markdown("---")
st.sidebar.markdown("[🔗 GitHub Repository](#)")

# Main Area
st.title("Posez votre question")

if not GOOGLE_API_KEY:
    st.error("⚠️ Clé API Google manquante. Veuillez vérifier votre fichier .env.")
elif doc_count == 0:
    st.warning("⚠️ La base de données est vide. Veuillez exécuter `python src/ingest.py` pour indexer vos PDF.")

question = st.text_input("Quelle est votre question sur les cours Smart ICT ?", placeholder="Ex: Qu'est-ce que la transformée de Fourier ?")

if st.button("Rechercher") and question:
    with st.spinner("Recherche d'informations et génération de la réponse..."):
        result = answer_question(question)
        
        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown("### Réponse")
            st.markdown(result["answer"])
            
            st.caption(f"⏱️ Temps de réponse : {result['latency_ms']} ms")
            
            with st.expander("🔍 Voir les sources"):
                for idx, source in enumerate(result["sources"]):
                    st.markdown(f"**Source {idx + 1}: {source['filename']} (Page {source['page_number']})**")
                    st.text(source.get("text", "Aperçu non disponible")[:300] + "...")
                    st.divider()
