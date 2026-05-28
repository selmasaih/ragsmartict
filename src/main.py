import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.query import answer_question, stream_answer, warmup_models, list_topics
from src.config import (
    CHROMA_DB_PATH, COLLECTION_NAME, CORS_ORIGINS, MAX_QUESTION_CHARS,
    LLM_PROVIDER, validate,
)
from src.logger import get_logger

log = get_logger("rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for problem in validate():
        log.warning("Config: %s", problem)
    warmup_models()
    yield


app = FastAPI(title="INPT Smart ICT Notes API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Optional[List[Dict[str, Any]]] = []
    topic: Optional[str] = None


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


@app.post("/api/query")
def query_rag(request: QueryRequest):
    question = _validate_question(request)
    result = answer_question(question, history=request.history, topic=request.topic)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/query/stream")
def query_rag_stream(request: QueryRequest):
    question = _validate_question(request)

    def event_stream():
        for event in stream_answer(question, history=request.history, topic=request.topic):
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
