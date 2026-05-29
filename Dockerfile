# Backend (FastAPI + RAG) image.
FROM python:3.11-slim

# System libs needed by PyMuPDF / OpenCV (easyocr) / PIL.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts

ENV PORT=8000
EXPOSE 8000

# Mount your notes/ and chroma_db/ as volumes, or run ingestion at build time.
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
