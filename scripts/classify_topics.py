"""Classify every document into a canonical Smart ICT topic and write a
reviewable TSV (topic_mapping.tsv). Nothing in the database is modified here —
review/edit the TSV, then run scripts/apply_topics.py.

The taxonomy and classifier live in src/topics.py (shared with ingestion).

Usage:  python -m scripts.classify_topics
"""
import os
import sys
import csv
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from src.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, BASE_DIR
from src.topics import classify_keywords, classify

OUTPUT = os.path.join(BASE_DIR, "topic_mapping.tsv")


def main():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()
    print(f"Scanning {total} chunks ...")

    folders = defaultdict(Counter)      # filename -> Counter(folder)
    samples = {}                        # filename -> text sample
    counts = Counter()                  # filename -> n chunks
    off = 0
    while off < total:
        data = col.get(include=["documents", "metadatas"], limit=5000, offset=off)
        for doc, meta in zip(data["documents"], data["metadatas"]):
            meta = meta or {}
            fn = meta.get("filename", "?")
            folders[fn][meta.get("subject", "?")] += 1
            counts[fn] += 1
            if fn not in samples and doc:
                samples[fn] = doc[:1500]
        off += 5000

    assignments = {}
    unmatched = []
    for fn in counts:
        folder = folders[fn].most_common(1)[0][0]
        topic, _ = classify_keywords(fn, folder, samples.get(fn, ""))
        if topic:
            assignments[fn] = topic
        else:
            unmatched.append(fn)

    if unmatched:
        print(f"Embedding fallback for {len(unmatched)} ambiguous files ...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        for fn in unmatched:
            folder = folders[fn].most_common(1)[0][0]
            assignments[fn] = classify(fn, folder, samples.get(fn, ""), model=model)

    rows = []
    for fn in counts:
        folder = folders[fn].most_common(1)[0][0]
        rows.append((assignments[fn], folder, fn, counts[fn]))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    with open(OUTPUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["topic", "original_folder", "filename", "n_chunks"])
        w.writerows(rows)

    dist = Counter(r[0] for r in rows)
    chunk_dist = Counter()
    for d, _, _, n in rows:
        chunk_dist[d] += n
    print(f"\nWrote {OUTPUT}  ({len(rows)} files)\n")
    print(f"{'TOPIC':<38} {'FILES':>6} {'CHUNKS':>8}")
    for d, _ in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"{d:<38} {dist[d]:>6} {chunk_dist[d]:>8}")


if __name__ == "__main__":
    main()
