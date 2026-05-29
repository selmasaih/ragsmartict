import os
import sys

import fitz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import topics, extract, ingest


# ── topics ───────────────────────────────────────────────────────────
def test_classify_keywords_known():
    assert topics.classify_keywords("Slides 3G_4G.pdf", "3G 4G", "")[0] == "Réseaux mobiles & radio"
    assert topics.classify_keywords("BD_INPT_2025_SQL.pdf", "Bases de données", "")[0] == "Informatique & données"


def test_classify_override_wins():
    assert topics.classify_keywords("IntroMATLAB.pdf", "Outils de simulation", "")[0] == topics.OTHER


def test_classify_fallback_without_model():
    # No keyword hit and no model -> OTHER.
    assert topics.classify("zzz.pdf", "qqq", "contenu sans signal", model=None) == topics.OTHER


def test_topics_taxonomy_has_eight():
    assert len(topics.TOPICS) == 8


# ── extract ──────────────────────────────────────────────────────────
def test_extract_any_unsupported_ext():
    assert extract.extract_any("notes.txt") == []


def test_image_exts_membership():
    assert ".png" in extract.IMAGE_EXTS and ".pdf" not in extract.IMAGE_EXTS


def test_extract_pdf_reads_text(tmp_path):
    p = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Bonjour ceci est un test RAG pour extraction de texte.")
    doc.save(str(p))
    doc.close()
    pages = extract.extract_pdf(str(p))
    assert pages and "extraction" in pages[0][1]


# ── ingest ───────────────────────────────────────────────────────────
import numpy as np


class _FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            self.seen.append(texts)
            return np.array([0.1, 0.2, 0.3])
        self.seen.extend(texts)
        return np.array([[0.1, 0.2, 0.3] for _ in texts])


def test_embed_passages_adds_prefix():
    m = _FakeModel()
    ingest.embed_passages(m, ["alpha", "beta"])
    assert all(t.startswith("passage: ") for t in m.seen)


def test_process_file_adds_chunks_and_topic(tmp_path):
    import chromadb
    p = tmp_path / "Slides 3G_4G.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Cours sur la 5G et le MIMO massif. " * 20)
    doc.save(str(p))
    doc.close()

    col = chromadb.Client().get_or_create_collection("test_col")
    n, topic = ingest.process_file(col, _FakeModel(), ingest.make_splitter(), str(p), "3G 4G")
    assert n > 0
    assert topic == "Réseaux mobiles & radio"
    assert col.count() == n


def test_process_file_dedup(tmp_path):
    import chromadb
    p = tmp_path / "dup.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Texte identique repete pour tester la deduplication exacte des chunks.")
    doc.save(str(p))
    doc.close()

    col = chromadb.Client().get_or_create_collection("dedup_col")
    seen = set()
    n1, _ = ingest.process_file(col, _FakeModel(), ingest.make_splitter(), str(p), "X", seen_hashes=seen)
    n2, _ = ingest.process_file(col, _FakeModel(), ingest.make_splitter(), str(p), "X", seen_hashes=seen)
    assert n1 > 0
    assert n2 == 0  # identical content skipped on second pass
