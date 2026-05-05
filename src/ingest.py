import os
import argparse
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.config import (
    NOTES_PATH, CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, COLLECTION_NAME
)

def ingest_documents(reset=False):
    if not os.path.exists(NOTES_PATH):
        os.makedirs(NOTES_PATH)
        print(f"Created directory {NOTES_PATH}. Please add your PDFs and run again.")
        return

    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    if reset:
        print(f"Resetting collection '{COLLECTION_NAME}'...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    pdf_files = []
    for root, _, files in os.walk(NOTES_PATH):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    if not pdf_files:
        print("No PDF files found in the notes/ directory.")
        return

    total_chunks = 0
    processed_pdfs = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        subject = os.path.basename(os.path.dirname(pdf_path))
        print(f"Processing {filename} (Subject: {subject})...")

        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or len(text.strip()) < 50:
                    continue

                chunks = text_splitter.split_text(text)
                if not chunks:
                    continue

                embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

                ids = [f"{filename}_p{page_num+1}_c{i}" for i in range(len(chunks))]
                metadatas = [
                    {
                        "filename": filename,
                        "subject": subject,
                        "page_number": page_num + 1,
                        "chunk_index": i,
                        "text": chunk
                    }
                    for i, chunk in enumerate(chunks)
                ]

                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas
                )
                total_chunks += len(chunks)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

        processed_pdfs += 1

    print(f"Ingestion complete. Processed {processed_pdfs}/{len(pdf_files)} PDFs, {total_chunks} chunks total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF notes into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Clear the collection before re-ingestion")
    args = parser.parse_args()
    ingest_documents(reset=args.reset)
