import os
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from src.query import (
    answer_question, stream_answer, warmup_models, list_topics,
    add_file_to_index, list_documents, delete_document,
)
from src.extract import IMAGE_EXTS, ocr_available
from src.config import (
    CHROMA_DB_PATH, COLLECTION_NAME, CORS_ORIGINS, MAX_QUESTION_CHARS,
    LLM_PROVIDER, NOTES_PATH, API_KEY, RATE_LIMIT, validate,
)
from src.logger import get_logger

UPLOAD_DIR = os.path.join(NOTES_PATH, "_uploads")
ALLOWED_UPLOAD_EXTS = {".pdf"} | IMAGE_EXTS
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

log = get_logger("rag.api")
limiter = Limiter(key_func=get_remote_address)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """No-op unless API_KEY is configured; then enforce X-API-Key header."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for problem in validate():
        log.warning("Config: %s", problem)
    warmup_models()
    yield


app = FastAPI(title="INPT Smart ICT Notes API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Optional[List[Dict[str, Any]]] = []
    topic: Optional[str] = None


class DeleteRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    subject: Optional[str] = None


def _validate_question(request: QueryRequest) -> str:
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="La question ne peut pas être vide.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"La question dépasse la limite de {MAX_QUESTION_CHARS} caractères.",
        )
    return question


@app.get("/api/health")
def health():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        count = client.get_collection(name=COLLECTION_NAME).count()
        status = "ok" if count > 0 else "empty"
    except Exception:
        count = 0
        status = "no_collection"
    return {"status": status, "doc_count": count, "llm_provider": LLM_PROVIDER}


@app.get("/api/stats")
def get_stats():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        return {"doc_count": collection.count()}
    except Exception:
        return {"doc_count": 0}


@app.get("/api/topics")
def get_topics():
    return {"topics": list_topics()}


@app.get("/api/documents")
def get_documents():
    return {"documents": list_documents()}


@app.delete("/api/documents", dependencies=[Depends(require_api_key)])
def remove_document(request: DeleteRequest):
    removed = delete_document(request.filename, subject=request.subject)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document introuvable.")
    return {"filename": request.filename, "chunks_removed": removed}


@app.post("/api/upload", dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
async def upload_file(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Type de fichier non supporté ({ext}). Formats acceptés : PDF, images.",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = os.path.basename(file.filename)
    dest = os.path.join(UPLOAD_DIR, safe_name)

    size = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                os.remove(dest)
                raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 25 Mo).")
            out.write(chunk)

    try:
        n, topic = add_file_to_index(dest, subject="Uploads")
    except Exception as e:
        log.error("Upload ingestion failed for %s: %s", safe_name, e)
        raise HTTPException(status_code=500, detail=f"Échec de l'indexation : {e}")

    if n == 0:
        hint = ""
        if not ocr_available():
            hint = " (OCR indisponible : impossible de lire les images/scans)."
        raise HTTPException(
            status_code=422,
            detail=f"Aucun texte exploitable extrait de ce fichier{hint} "
                   "Le PDF est peut-être protégé par mot de passe ou vide.",
        )

    return {"filename": safe_name, "chunks_added": n, "topic": topic}


@app.post("/api/query", dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
def query_rag(request: Request, body: QueryRequest):
    question = _validate_question(body)
    result = answer_question(question, history=body.history, topic=body.topic)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/query/stream", dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
def query_rag_stream(request: Request, body: QueryRequest):
    question = _validate_question(body)

    def event_stream():
        for event in stream_answer(question, history=body.history, topic=body.topic):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "0").strip().lower() in {"1", "true", "yes"}
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=reload)
