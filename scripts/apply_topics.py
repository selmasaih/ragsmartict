"""Apply the reviewed topic_mapping.tsv to the ChromaDB collection.

Reads topic_mapping.tsv (topic / original_folder / filename / n_chunks) and
writes a `topic` field into every matching chunk's metadata. Match is done on
(filename, original_folder) so duplicate filenames in different folders are
handled correctly.

Usage:  python -m scripts.apply_topics
"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from src.config import CHROMA_DB_PATH, COLLECTION_NAME, BASE_DIR

MAPPING = os.path.join(BASE_DIR, "topic_mapping.tsv")


def main():
    if not os.path.exists(MAPPING):
        print(f"Mapping file not found: {MAPPING}\nRun scripts/classify_topics.py first.")
        return

    with open(MAPPING, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_collection(COLLECTION_NAME)

    updated = 0
    for i, row in enumerate(rows, 1):
        topic = row["topic"].strip()
        folder = row["original_folder"]
        filename = row["filename"]
        where = {"$and": [{"filename": {"$eq": filename}}, {"subject": {"$eq": folder}}]}
        data = col.get(where=where, include=["metadatas"])
        ids = data["ids"]
        if not ids:
            continue
        metas = data["metadatas"]
        for m in metas:
            m["topic"] = topic
        # Update in batches to stay well under any payload limits.
        for j in range(0, len(ids), 1000):
            col.update(ids=ids[j:j + 1000], metadatas=metas[j:j + 1000])
        updated += len(ids)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} files processed ...")

    print(f"Done. Tagged {updated} chunks with a `topic` across {len(rows)} files.")


if __name__ == "__main__":
    main()
