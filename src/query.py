import re
import time
import requests
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import (
    CHROMA_DB_PATH, EMBEDDING_MODEL,
    OLLAMA_MODEL, OLLAMA_API_URL, COLLECTION_NAME, TOP_K,
    OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT_S,
    OLLAMA_NUM_PREDICT, OLLAMA_TEMPERATURE, OLLAMA_TOP_P, OLLAMA_TOP_K,
    CONTEXT_MAX_CHARS, CONTEXT_MAX_CHUNK_CHARS,
    VECTOR_K, BM25_K, RERANK_TOP_K, RERANKER_MODEL,
    ENABLE_QUERY_REWRITE, REWRITE_MAX_WORDS,
    BM25_PAGE_SIZE, BM25_MAX_DOCS,
    LLM_PROVIDER, GOOGLE_API_KEY
)

import google.generativeai as genai

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


_EMBED_MODEL = None
_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None
_BM25_INDEX = None
_BM25_DOCS = None
_BM25_METAS = None
_RERANKER = None


def _get_embedding_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBED_MODEL


def _get_collection():
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        import os
        print(f"[RAG] Connecting to ChromaDB at: {CHROMA_DB_PATH} (exists={os.path.exists(CHROMA_DB_PATH)})")
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _CHROMA_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
        count = _CHROMA_COLLECTION.count()
        print(f"[RAG] Collection '{COLLECTION_NAME}' loaded — {count} chunks")
        if count == 0:
            print("[RAG] WARNING: Collection is empty. Run ingestion first.")
    return _CHROMA_COLLECTION


def _tokenize(text: str):
    if not text:
        return []
    return re.findall(r"[\w']+", text.lower())


def _extractive_answer(chunks, max_sentences: int = 2, max_chars: int = 500) -> str:
    if not chunks:
        return ""
    text = (chunks[0] or "").strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    extract = " ".join(sentences[:max_sentences]).strip()
    if not extract:
        extract = text[:max_chars].strip()
    return extract


def _looks_like_reasoning(text: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    cues = [
        "the user",
        "instructions",
        "format",
        "i need to",
        "let's",
        "wait,",
    ]
    return any(cue in lower for cue in cues)


def _make_candidate_id(meta, fallback: str) -> str:
    meta = meta or {}
    filename = meta.get("filename")
    page_number = meta.get("page_number")
    chunk_index = meta.get("chunk_index")
    if filename is not None and page_number is not None:
        return f"{filename}|{page_number}|{chunk_index}"
    return fallback


def _build_system_prompt():
    return (
        "Tu es un assistant pour un etudiant ingenieur a INPT Rabat (filiere Smart ICT). "
        "Adopte un ton professionnel, clair et concis. "
        "Structure la reponse en paragraphes courts; utilise des puces si cela clarifie. "
        "Reponds directement, sans raisonnement interne ni balises <think>. "
        "Garde la reponse courte (max 120 mots). "
        "Reponds UNIQUEMENT en francais et UNIQUEMENT avec les informations des extraits fournis. "
        "Si l'information n'est pas dans les extraits, dis clairement: 'Je ne trouve pas cette information dans tes notes.' "
        "Ne cite pas les sources dans la reponse; elles seront affichees separement. "
        "N'utilise pas de references inline comme [fichier, page]."
    )


def _call_ollama(prompt: str, system_prompt: str = "") -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
        },
    }

    response = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT_S)
    response.raise_for_status()
    raw = response.json().get("response", "")

    # Strip <think>...</think> reasoning block from thinking models
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "<think>" in raw and not cleaned:
        # If <think> never closed, drop everything from it onward
        cleaned = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
        return cleaned
    return cleaned if cleaned else raw


def _call_gemini(prompt: str, system_prompt: str = "") -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = f"Instructions système:\n{system_prompt}\n\nQuestion et contexte:\n{prompt}" if system_prompt else prompt
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[RAG] Gemini error: {e}")
        raise e


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANKER_MODEL)
    return _RERANKER


def _should_rewrite(question: str) -> bool:
    if not ENABLE_QUERY_REWRITE:
        return False
    word_count = len(question.strip().split())
    return 0 < word_count <= REWRITE_MAX_WORDS


def _rewrite_query(question: str) -> str:
    if not _should_rewrite(question):
        return question
    try:
        system_instruction = (
            "Tu es un assistant de recherche. Reformule la question en une requete courte et claire, "
            "sans ajouter d'informations. Reponds par une seule ligne, sans guillemets."
        )
        if LLM_PROVIDER.lower() == "gemini":
            rewritten = _call_gemini(question, system_prompt=system_instruction).strip().splitlines()[0].strip()
        else:
            rewritten = _call_ollama(question, system_prompt=system_instruction).strip().splitlines()[0].strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def _get_bm25_index(collection):
    global _BM25_INDEX, _BM25_DOCS, _BM25_METAS
    if _BM25_INDEX is None:
        total = collection.count()
        if total <= 0:
            return None

        max_docs = min(total, BM25_MAX_DOCS) if BM25_MAX_DOCS else total
        page_size = max(1, BM25_PAGE_SIZE)
        docs = []
        metas = []
        offset = 0

        while offset < max_docs:
            limit = min(page_size, max_docs - offset)
            data = collection.get(include=["documents", "metadatas"], limit=limit, offset=offset)
            page_docs = data.get("documents", []) or []
            page_metas = data.get("metadatas", []) or []
            if not page_docs:
                break

            for doc, meta in zip(page_docs, page_metas):
                if not doc:
                    continue
                docs.append(doc)
                metas.append(meta)

            offset += len(page_docs)

        _BM25_DOCS = docs
        _BM25_METAS = metas
        tokenized = [_tokenize(doc) for doc in _BM25_DOCS]
        _BM25_INDEX = BM25Okapi(tokenized) if tokenized else None
    return _BM25_INDEX


def _build_sources(retrieved_chunks, metadatas):
    sources = []
    seen = set()
    for chunk, meta in zip(retrieved_chunks, metadatas):
        meta = meta or {}
        filename = meta.get("filename", "Document")
        page_number = meta.get("page_number", "N/A")
        key = (filename, page_number)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "filename": filename,
            "page_number": page_number,
            "text": chunk,
        })
    return sources


def _collect_vector_candidates(results):
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates = []
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        if not doc:
            continue
        score = 1 / (1 + dist) if dist is not None else 0
        doc_id = _make_candidate_id(meta, f"vec_{idx}")
        candidates.append({
            "id": doc_id,
            "doc": doc,
            "meta": meta,
            "score": score,
        })
    return candidates


def _collect_bm25_candidates(collection, query_text: str):
    try:
        index = _get_bm25_index(collection)
    except Exception:
        return []
    if index is None or not _BM25_DOCS:
        return []
    scores = index.get_scores(_tokenize(query_text))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_K]
    candidates = []
    for idx in ranked:
        meta = _BM25_METAS[idx]
        doc_id = _make_candidate_id(meta, f"bm25_{idx}")
        candidates.append({
            "id": doc_id,
            "doc": _BM25_DOCS[idx],
            "meta": meta,
            "score": scores[idx],
        })
    return candidates


def _merge_candidates(*candidate_lists):
    merged = {}
    for candidates in candidate_lists:
        for candidate in candidates:
            doc_id = candidate["id"]
            if doc_id not in merged or candidate["score"] > merged[doc_id]["score"]:
                merged[doc_id] = candidate
    return list(merged.values())


def _rerank_candidates(query_text: str, candidates):
    if not candidates:
        return [], []
    reranker = _get_reranker()
    pairs = [(query_text, c["doc"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top = ranked[:RERANK_TOP_K]
    retrieved_chunks = [candidates[i]["doc"] for i in top]
    metadatas = [candidates[i]["meta"] for i in top]
    return retrieved_chunks, metadatas


# ── Pre-warm all heavy models at startup ────────────────────────────
def warmup_models():
    """Pre-load embedding model, reranker, ChromaDB collection, and BM25
    index so the first user query isn't penalised by cold-start latency."""
    print("[RAG] Pre-warming models ...")
    t0 = time.time()

    t = time.time()
    _get_embedding_model()
    print(f"[RAG]   OK Embedding model loaded       ({time.time() - t:.1f}s)")

    t = time.time()
    _get_reranker()
    print(f"[RAG]   OK Reranker loaded               ({time.time() - t:.1f}s)")

    t = time.time()
    col = _get_collection()
    print(f"[RAG]   OK ChromaDB collection loaded    ({time.time() - t:.1f}s)")

    t = time.time()
    _get_bm25_index(col)
    print(f"[RAG]   OK BM25 index built              ({time.time() - t:.1f}s)")

    # Warm LLM by sending a tiny prompt so the model is loaded in memory
    t = time.time()
    try:
        if LLM_PROVIDER.lower() == "gemini":
            _call_gemini("ping", system_prompt="Reply with 'pong' only.")
            print(f"[RAG]   OK Gemini model warmed          ({time.time() - t:.1f}s)")
        else:
            _call_ollama("ping", system_prompt="Reply with 'pong' only.")
            print(f"[RAG]   OK Ollama model warmed          ({time.time() - t:.1f}s)")
    except Exception as e:
        print(f"[RAG]   WARN LLM warm-up failed: {e}")

    print(f"[RAG] All models ready in {time.time() - t0:.1f}s")


def answer_question(question: str) -> dict:
    start_time = time.time()
    timings = {}

    try:
        collection = _get_collection()
        if collection.count() == 0:
            return {
                "error": "La base de données vectorielle est vide. Veuillez lancer l'ingestion d'abord avec: python -m src.ingest",
                "sources": [], "latency_ms": 0
            }
    except Exception as e:
        import traceback
        err_msg = (
            f"Erreur d'accès à la base de données vectorielle.\n"
            f"Chemin: {CHROMA_DB_PATH}\n"
            f"Exception: {str(e)}\n"
            f"{traceback.format_exc()}"
        )
        print(f"[RAG] ERROR: {err_msg}")
        return {"error": err_msg, "sources": [], "latency_ms": 0}

    # 1. Optionally rewrite the query for retrieval
    t = time.time()
    query_text = _rewrite_query(question)
    timings["query_rewrite_ms"] = int((time.time() - t) * 1000)

    # 2. Embed the query
    t = time.time()
    model = _get_embedding_model()
    question_embedding = model.encode(query_text, normalize_embeddings=True).tolist()
    timings["embedding_ms"] = int((time.time() - t) * 1000)

    # 3. Vector retrieval
    t = time.time()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=max(TOP_K, VECTOR_K),
        include=["documents", "metadatas", "distances"]
    )
    vector_candidates = _collect_vector_candidates(results)
    timings["vector_retrieval_ms"] = int((time.time() - t) * 1000)

    # 4. BM25 retrieval
    t = time.time()
    bm25_candidates = _collect_bm25_candidates(collection, query_text)
    timings["bm25_retrieval_ms"] = int((time.time() - t) * 1000)

    # 5. Merge candidates
    t = time.time()
    candidates = _merge_candidates(vector_candidates, bm25_candidates)
    timings["merge_ms"] = int((time.time() - t) * 1000)

    if not candidates:
        return {"answer": "Aucun document pertinent trouvé dans la base de données.", "sources": [], "latency_ms": int((time.time() - start_time) * 1000)}

    # 6. Rerank
    t = time.time()
    try:
        retrieved_chunks, metadatas = _rerank_candidates(query_text, candidates)
    except Exception:
        candidates.sort(key=lambda c: c["score"], reverse=True)
        top = candidates[:TOP_K]
        retrieved_chunks = [c["doc"] for c in top]
        metadatas = [c["meta"] for c in top]
    timings["rerank_ms"] = int((time.time() - t) * 1000)

    context_blocks = []
    used_chunks = []
    used_metas = []
    total_chars = 0

    for chunk, meta in zip(retrieved_chunks, metadatas):
        if not chunk:
            continue
        snippet = chunk.strip()
        if CONTEXT_MAX_CHUNK_CHARS and len(snippet) > CONTEXT_MAX_CHUNK_CHARS:
            snippet = snippet[:CONTEXT_MAX_CHUNK_CHARS].rstrip() + "…"

        block = f"Extrait:\n{snippet}"
        next_total = total_chars + len(block) + 1
        if CONTEXT_MAX_CHARS and next_total > CONTEXT_MAX_CHARS:
            break

        context_blocks.append(block)
        used_chunks.append(chunk)
        used_metas.append(meta)
        total_chars = next_total
    
    context_str = "\n".join(context_blocks)

    sources = _build_sources(used_chunks, used_metas)

    # 7. Build user message
    user_message = f"Contexte:\n{context_str}\n\nQuestion: {question}"

    # 8. Call LLM (Gemini or Ollama)
    t = time.time()
    try:
        if LLM_PROVIDER.lower() == "gemini":
            answer = _call_gemini(user_message, system_prompt=_build_system_prompt())
        else:
            answer = _call_ollama(user_message, system_prompt=_build_system_prompt())
    except Exception as e:
        error_text = str(e)
        if "Connection" in error_text or "ConnectionRefusedError" in error_text:
            fallback = (
                "Le service LLM est inaccessible. Assurez-vous qu'il est configuré "
                "et lancé correctement."
            )
        else:
            fallback = _extractive_answer(retrieved_chunks) or (
                "Je ne peux pas generer la reponse pour le moment. Reessayez plus tard."
            )
        return {
            "answer": fallback,
            "sources": sources,
            "latency_ms": int((time.time() - start_time) * 1000),
            "warning": error_text,
        }
    timings["llm_generation_ms"] = int((time.time() - t) * 1000)

    latency_ms = int((time.time() - start_time) * 1000)

    # Log per-step timings
    print(f"[RAG] Timings: {timings}  |  Total: {latency_ms}ms")

    if _looks_like_reasoning(answer):
        answer = _extractive_answer(retrieved_chunks) or answer

    # 9. Return response
    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency_ms,
        "timings": timings,
    }
