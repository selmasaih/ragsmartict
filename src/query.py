import re
import time
import json
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
    VECTOR_K, BM25_K, RERANK_TOP_K, RERANKER_MODEL, ENABLE_RERANK,
    ENABLE_QUERY_REWRITE, REWRITE_MAX_WORDS,
    BM25_PAGE_SIZE, BM25_MAX_DOCS, MAX_HISTORY_MESSAGES,
    LLM_PROVIDER, GOOGLE_API_KEY, GEMINI_MODEL,
)
from src.logger import get_logger

import google.generativeai as genai

log = get_logger("rag.query")

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
        log.info("Connecting to ChromaDB at %s (exists=%s)", CHROMA_DB_PATH, os.path.exists(CHROMA_DB_PATH))
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _CHROMA_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
        count = _CHROMA_COLLECTION.count()
        log.info("Collection '%s' loaded — %d chunks", COLLECTION_NAME, count)
        if count == 0:
            log.warning("Collection is empty. Run ingestion first.")
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
        "Tu es un assistant expert pour un etudiant ingenieur a INPT Rabat (filiere Smart ICT). "
        "Adopte un ton professionnel, précis et scientifique. "
        "Structure la reponse en paragraphes courts ou listes à puces. "
        "REVISE TES SOURCES : Avant de repondre, verifie si les informations sont coherentes. "
        "Si deux extraits semblent se contredire (ex: differentes annees ou matieres), precise-le. "
        "NE HALUCINE PAS : Si l'information exacte n'est pas dans les extraits, dis 'Information non trouvee'. "
        "Reponds UNIQUEMENT en francais et UNIQUEMENT avec les extraits fournis. "
        "Ne cite pas les sources dans le texte (ex: [1]); elles sont listees en dessous."
    )


# ── Think-tag stripping (streaming-safe) ─────────────────────────────
def _make_think_filter():
    """Return (feed, flush) closures that suppress <think>...</think> blocks
    from a token stream, handling tags split across chunks."""
    OPEN, CLOSE = "<think>", "</think>"
    state = {"buf": "", "in_think": False}

    def feed(text: str = "") -> str:
        state["buf"] += text
        out = []
        while True:
            if state["in_think"]:
                i = state["buf"].find(CLOSE)
                if i == -1:
                    keep = len(CLOSE) - 1
                    if len(state["buf"]) > keep:
                        state["buf"] = state["buf"][-keep:]
                    break
                state["buf"] = state["buf"][i + len(CLOSE):]
                state["in_think"] = False
            else:
                i = state["buf"].find(OPEN)
                if i == -1:
                    keep = len(OPEN) - 1
                    if len(state["buf"]) > keep:
                        out.append(state["buf"][:-keep])
                        state["buf"] = state["buf"][-keep:]
                    break
                out.append(state["buf"][:i])
                state["buf"] = state["buf"][i + len(OPEN):]
                state["in_think"] = True
        return "".join(out)

    def flush() -> str:
        if state["in_think"]:
            state["buf"] = ""
            return ""
        out = state["buf"]
        state["buf"] = ""
        return out

    return feed, flush


def _strip_think_blocking(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Drop any leftover unclosed think block (model truncated mid-reasoning).
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    if cleaned:
        return cleaned
    # Nothing left after stripping: only fall back to raw if there was no think tag.
    return "" if "<think>" in raw else raw


def _ollama_payload(prompt: str, system_prompt: str, stream: bool) -> dict:
    return {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": stream,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
        },
    }


def _call_ollama(prompt: str, system_prompt: str = "") -> str:
    response = requests.post(
        OLLAMA_API_URL, json=_ollama_payload(prompt, system_prompt, False), timeout=OLLAMA_TIMEOUT_S
    )
    response.raise_for_status()
    raw = response.json().get("response", "")
    return _strip_think_blocking(raw)


def _stream_ollama(prompt: str, system_prompt: str = ""):
    feed, flush = _make_think_filter()
    with requests.post(
        OLLAMA_API_URL, json=_ollama_payload(prompt, system_prompt, True),
        timeout=OLLAMA_TIMEOUT_S, stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = data.get("response", "")
            if piece:
                visible = feed(piece)
                if visible:
                    yield visible
            if data.get("done"):
                break
    tail = flush()
    if tail:
        yield tail


def _call_gemini(prompt: str, system_prompt: str = "") -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)
    full_prompt = (
        f"Instructions système:\n{system_prompt}\n\nQuestion et contexte:\n{prompt}"
        if system_prompt else prompt
    )
    response = model.generate_content(full_prompt)
    return response.text.strip()


def _stream_gemini(prompt: str, system_prompt: str = ""):
    model = genai.GenerativeModel(GEMINI_MODEL)
    full_prompt = (
        f"Instructions système:\n{system_prompt}\n\nQuestion et contexte:\n{prompt}"
        if system_prompt else prompt
    )
    for chunk in model.generate_content(full_prompt, stream=True):
        text = getattr(chunk, "text", "")
        if text:
            yield text


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
        if LLM_PROVIDER == "gemini":
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


def add_file_to_index(path: str, subject: str = "Uploads"):
    """Ingest a single uploaded file into the live collection and invalidate
    the BM25 cache so it is immediately searchable. Returns (chunks_added, topic)."""
    global _BM25_INDEX, _BM25_DOCS, _BM25_METAS
    from src.ingest import process_file, make_splitter
    collection = _get_collection()
    model = _get_embedding_model()
    n, topic = process_file(collection, model, make_splitter(), path, subject)
    _BM25_INDEX = None
    _BM25_DOCS = None
    _BM25_METAS = None
    log.info("Uploaded '%s' -> topic=%s (%d chunks)", path, topic, n)
    return n, topic


def list_topics() -> list[str]:
    """Return the distinct canonical topics present in the collection."""
    try:
        collection = _get_collection()
        _get_bm25_index(collection)  # populates _BM25_METAS cheaply
        metas = _BM25_METAS or []
        topics = {m.get("topic") for m in metas if m and m.get("topic")}
        return sorted(topics)
    except Exception:
        return []


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
            "subject": meta.get("subject"),
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
        candidates.append({"id": doc_id, "doc": doc, "meta": meta, "score": score})
    return candidates


def _collect_bm25_candidates(collection, query_text: str, topic: str = None):
    try:
        index = _get_bm25_index(collection)
    except Exception:
        return []
    if index is None or not _BM25_DOCS:
        return []
    scores = index.get_scores(_tokenize(query_text))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    candidates = []
    for idx in order:
        meta = _BM25_METAS[idx]
        if topic and (meta or {}).get("topic") != topic:
            continue
        doc_id = _make_candidate_id(meta, f"bm25_{idx}")
        candidates.append({"id": doc_id, "doc": _BM25_DOCS[idx], "meta": meta, "score": scores[idx]})
        if len(candidates) >= BM25_K:
            break
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
    log.info("Pre-warming models ...")
    t0 = time.time()

    t = time.time()
    _get_embedding_model()
    log.info("  OK Embedding model loaded (%.1fs)", time.time() - t)

    if ENABLE_RERANK:
        t = time.time()
        _get_reranker()
        log.info("  OK Reranker loaded (%.1fs)", time.time() - t)
    else:
        log.info("  SKIP Reranker (disabled)")

    t = time.time()
    col = _get_collection()
    log.info("  OK ChromaDB collection loaded (%.1fs)", time.time() - t)

    if BM25_K > 0:
        t = time.time()
        _get_bm25_index(col)
        log.info("  OK BM25 index built (%.1fs)", time.time() - t)
    else:
        log.info("  SKIP BM25 index (disabled)")

    t = time.time()
    try:
        if LLM_PROVIDER == "gemini":
            _call_gemini("ping", system_prompt="Reply with 'pong' only.")
            log.info("  OK Gemini model warmed (%.1fs)", time.time() - t)
        else:
            _call_ollama("ping", system_prompt="Reply with 'pong' only.")
            log.info("  OK Ollama model warmed (%.1fs)", time.time() - t)
    except Exception as e:
        log.warning("  LLM warm-up failed: %s", e)

    log.info("All models ready in %.1fs", time.time() - t0)


def _ensure_collection_ready():
    """Return (collection, error_dict). error_dict is None when ready."""
    try:
        collection = _get_collection()
    except Exception as e:
        import traceback
        err_msg = (
            f"Erreur d'accès à la base de données vectorielle.\n"
            f"Chemin: {CHROMA_DB_PATH}\nException: {e}\n{traceback.format_exc()}"
        )
        log.error(err_msg)
        return None, {"error": err_msg, "sources": [], "latency_ms": 0}
    if collection.count() == 0:
        return None, {
            "error": "La base de données vectorielle est vide. Veuillez lancer l'ingestion d'abord avec: python -m src.ingest",
            "sources": [], "latency_ms": 0,
        }
    return collection, None


def _retrieve(collection, question: str, topic: str = None):
    """Run the full retrieval pipeline. Returns (context_str, sources, retrieved_chunks, timings)."""
    timings = {}

    t = time.time()
    query_text = _rewrite_query(question)
    timings["query_rewrite_ms"] = int((time.time() - t) * 1000)

    t = time.time()
    model = _get_embedding_model()
    # multilingual-e5 expects a "query: " prefix on search queries.
    question_embedding = model.encode("query: " + query_text, normalize_embeddings=True).tolist()
    timings["embedding_ms"] = int((time.time() - t) * 1000)

    t = time.time()
    import concurrent.futures
    where = {"topic": topic} if topic else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        def fetch_vector():
            res = collection.query(
                query_embeddings=[question_embedding],
                n_results=max(TOP_K, VECTOR_K),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return _collect_vector_candidates(res)

        def fetch_bm25():
            if BM25_K > 0:
                return _collect_bm25_candidates(collection, query_text, topic=topic)
            return []

        future_vector = executor.submit(fetch_vector)
        future_bm25 = executor.submit(fetch_bm25)
        vector_candidates = future_vector.result()
        bm25_candidates = future_bm25.result()
    timings["parallel_retrieval_ms"] = int((time.time() - t) * 1000)

    t = time.time()
    candidates = _merge_candidates(vector_candidates, bm25_candidates)
    timings["merge_ms"] = int((time.time() - t) * 1000)

    if not candidates:
        return "", [], [], timings

    t = time.time()
    if ENABLE_RERANK:
        try:
            retrieved_chunks, metadatas = _rerank_candidates(query_text, candidates)
        except Exception:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            top = candidates[:TOP_K]
            retrieved_chunks = [c["doc"] for c in top]
            metadatas = [c["meta"] for c in top]
    else:
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
        fname = (meta or {}).get("filename", "Inconnu")
        pnum = (meta or {}).get("page_number", "?")
        block = f"--- SOURCE: {fname} (Page {pnum}) ---\n{snippet}"
        next_total = total_chars + len(block) + 2
        if CONTEXT_MAX_CHARS and next_total > CONTEXT_MAX_CHARS:
            break
        context_blocks.append(block)
        used_chunks.append(chunk)
        used_metas.append(meta)
        total_chars = next_total

    context_str = "\n\n".join(context_blocks)
    sources = _build_sources(used_chunks, used_metas)
    return context_str, sources, used_chunks, timings


def _build_user_message(context_str: str, history: list, question: str) -> str:
    history_str = ""
    if history:
        history_str = "Historique de la conversation:\n"
        for msg in history[-MAX_HISTORY_MESSAGES:]:
            role = "Utilisateur" if msg.get("role") == "user" else "Assistant"
            history_str += f"{role}: {msg.get('content')}\n"
        history_str += "\n"
    return f"{history_str}Contexte:\n{context_str}\n\nQuestion: {question}"


def answer_question(question: str, history: list = None, topic: str = None) -> dict:
    """Non-streaming RAG answer."""
    start_time = time.time()
    history = history or []

    collection, err = _ensure_collection_ready()
    if err:
        return err

    context_str, sources, retrieved_chunks, timings = _retrieve(collection, question, topic)
    if not retrieved_chunks:
        return {
            "answer": "Aucun document pertinent trouvé dans la base de données.",
            "sources": [], "latency_ms": int((time.time() - start_time) * 1000),
        }

    user_message = _build_user_message(context_str, history, question)

    t = time.time()
    try:
        if LLM_PROVIDER == "gemini":
            answer = _call_gemini(user_message, system_prompt=_build_system_prompt())
        else:
            answer = _call_ollama(user_message, system_prompt=_build_system_prompt())
    except Exception as e:
        error_text = str(e)
        if "Connection" in error_text or "ConnectionRefusedError" in error_text:
            fallback = "Le service LLM est inaccessible. Assurez-vous qu'il est configuré et lancé correctement."
        else:
            fallback = _extractive_answer(retrieved_chunks) or (
                "Je ne peux pas generer la reponse pour le moment. Reessayez plus tard."
            )
        log.warning("LLM generation failed: %s", error_text)
        return {
            "answer": fallback,
            "sources": sources,
            "latency_ms": int((time.time() - start_time) * 1000),
            "warning": error_text,
        }
    timings["llm_generation_ms"] = int((time.time() - t) * 1000)

    latency_ms = int((time.time() - start_time) * 1000)
    log.info("Timings: %s | Total: %dms", timings, latency_ms)
    return {"answer": answer, "sources": sources, "latency_ms": latency_ms, "timings": timings}


def stream_answer(question: str, history: list = None, topic: str = None):
    """Generator yielding event dicts for SSE:
       {"type": "token", "text": ...}
       {"type": "sources", "sources": [...]}
       {"type": "done", "latency_ms": ...}
       {"type": "error", "error": ...}
    """
    start_time = time.time()
    history = history or []

    collection, err = _ensure_collection_ready()
    if err:
        yield {"type": "error", "error": err["error"]}
        return

    context_str, sources, retrieved_chunks, timings = _retrieve(collection, question, topic)
    if not retrieved_chunks:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": "Aucun document pertinent trouvé dans la base de données."}
        yield {"type": "done", "latency_ms": int((time.time() - start_time) * 1000)}
        return

    # Send sources up front so the UI can render citations immediately.
    yield {"type": "sources", "sources": sources}

    user_message = _build_user_message(context_str, history, question)
    system_prompt = _build_system_prompt()

    produced_any = False
    try:
        generator = _stream_gemini if LLM_PROVIDER == "gemini" else _stream_ollama
        for piece in generator(user_message, system_prompt=system_prompt):
            produced_any = True
            yield {"type": "token", "text": piece}
    except Exception as e:
        error_text = str(e)
        log.warning("Streaming generation failed: %s", error_text)
        if not produced_any:
            if "Connection" in error_text or "ConnectionRefusedError" in error_text:
                fallback = "Le service LLM est inaccessible. Assurez-vous qu'il est configuré et lancé correctement."
            else:
                fallback = _extractive_answer(retrieved_chunks) or (
                    "Je ne peux pas generer la reponse pour le moment. Reessayez plus tard."
                )
            yield {"type": "token", "text": fallback}
        yield {"type": "error", "error": error_text}

    latency_ms = int((time.time() - start_time) * 1000)
    log.info("Stream timings: %s | Total: %dms", timings, latency_ms)
    yield {"type": "done", "latency_ms": latency_ms}
