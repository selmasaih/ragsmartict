#!/bin/bash
echo "Starting FastAPI (RAG) on Azure App Service..."
python -m pip install -r requirements.txt
python -m uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8000}"
