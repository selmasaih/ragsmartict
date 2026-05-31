import os

# Prevent thread/process oversubscription that can livelock ingestion on
# Windows (set before importing torch/sentence-transformers).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import argparse
import multiprocessing
import chromadb

# PDF text can contain glyphs the Windows console (cp1252) cannot encode;
# make stdout/stderr tolerant so logging never crashes ingestion.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.config import (
    NOTES_PATH, CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, COLLECTION_NAME
)
from src.topics import classify
from src.extract import extract_any, IMAGE_EXTS

# multilingual-e5 expects a "passage: " prefix on indexed documents and a
# "query: " prefix on search queries for best retrieval quality.
PASSAGE_PREFIX = "passage: "


def make_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def embed_passages(model, chunks):
    return model.encode([PASSAGE_PREFIX + c for c in chunks], normalize_embeddings=True).tolist()


def process_file(collection, model, text_splitter, path, subject, seen_hashes=None):
    """Extract, chunk, classify, embed and add one file to the collection.
    Skips chunks whose exact content was already seen (when seen_hashes is
    provided). Returns (chunks_added, topic)."""
    import hashlib
    import unicodedata
    # Normalize to NFC so filenames with accents (é, etc.) match consistently
    # regardless of the source OS's Unicode normalization (NFC vs NFD).
    filename = unicodedata.normalize("NFC", os.path.basename(path))
    all_chunks, all_ids, all_metadatas = [], [], []

    for page_num, text in extract_any(path):
        if not text or len(text.strip()) < 50:
            continue
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            if seen_hashes is not None:
                h = hashlib.md5(chunk.strip().encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
            all_chunks.append(chunk)
            all_ids.append(f"{subject}|{filename}|p{page_num}|c{i}")
            all_metadatas.append({
                "filename": filename,
                "subject": subject,
                "page_number": page_num,
                "chunk_index": i,
            })

    if not all_chunks:
        return 0, None

    sample = " ".join(all_chunks[:2])[:1500]
    topic = classify(filename, subject, sample, model=model)
    for meta in all_metadatas:
        meta["topic"] = topic

    embeddings = embed_passages(model, all_chunks)
    # upsert (not add) so re-uploading the same file replaces its chunks
    # instead of raising on duplicate deterministic IDs.
    collection.upsert(ids=all_ids, embeddings=embeddings, documents=all_chunks, metadatas=all_metadatas)
    return len(all_chunks), topic


def ingest_documents(reset=False):
    if not os.path.exists(NOTES_PATH):
        os.makedirs(NOTES_PATH)
        print(f"Created directory {NOTES_PATH}. Please add your files and run again.")
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
    try:
        import torch
        torch.set_num_threads(4)
    except Exception:
        pass
    text_splitter = make_splitter()

    supported = {".pdf"} | IMAGE_EXTS
    files = []
    for root, _, names in os.walk(NOTES_PATH):
        for name in names:
            if os.path.splitext(name)[1].lower() in supported:
                files.append(os.path.join(root, name))

    if not files:
        print("No PDF/image files found in the notes/ directory.")
        return

    total_chunks = 0
    processed = 0
    seen_hashes = set()  # corpus-wide exact-duplicate chunk dedup
    for path in files:
        filename = os.path.basename(path)
        subject = os.path.basename(os.path.dirname(path))
        print(f"Processing {filename} (Subject: {subject})...")
        try:
            n, topic = process_file(collection, model, text_splitter, path, subject, seen_hashes=seen_hashes)
            if n:
                print(f"  -> topic: {topic} ({n} chunks)")
            total_chunks += n
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        processed += 1

    print(f"Ingestion complete. Processed {processed}/{len(files)} files, {total_chunks} chunks total.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Ingest PDF/image notes into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Clear the collection before re-ingestion")
    args = parser.parse_args()
    ingest_documents(reset=args.reset)
