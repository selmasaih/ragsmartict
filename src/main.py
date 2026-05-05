import time
import chromadb
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.query import answer_question, warmup_models
from src.config import CHROMA_DB_PATH, COLLECTION_NAME


# ── Lifespan: pre-warm all models before accepting requests ─────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_models()
    yield


app = FastAPI(title="INPT Smart ICT Notes API", lifespan=lifespan)

# Configure CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/api/stats")
def get_stats():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        return {"doc_count": count}
    except Exception:
        return {"doc_count": 0}

@app.post("/api/query")
def query_rag(request: QueryRequest):
    result = answer_question(request.question)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "0").strip().lower() in {"1", "true", "yes"}
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=reload)
