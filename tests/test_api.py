import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import main


@pytest.fixture
def client(monkeypatch):
    # Avoid loading heavy models / pinging the LLM during app startup.
    monkeypatch.setattr(main, "warmup_models", lambda: None)
    with TestClient(main.app) as c:
        yield c


def test_health_ok(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body and "doc_count" in body and "llm_provider" in body


def test_topics(client, monkeypatch):
    monkeypatch.setattr(main, "list_topics", lambda: ["Maths & optimisation", "Autres"])
    res = client.get("/api/topics")
    assert res.status_code == 200
    assert res.json()["topics"] == ["Maths & optimisation", "Autres"]


def test_query_empty_question_rejected(client):
    res = client.post("/api/query", json={"question": "   "})
    assert res.status_code == 422


def test_query_too_long_rejected(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_QUESTION_CHARS", 10)
    res = client.post("/api/query", json={"question": "x" * 50})
    assert res.status_code == 422


def test_query_happy_path(client, monkeypatch):
    monkeypatch.setattr(
        main, "answer_question",
        lambda q, history=None, topic=None, filename=None: {"answer": "ok", "sources": [], "latency_ms": 1},
    )
    res = client.post("/api/query", json={"question": "Bonjour ?"})
    assert res.status_code == 200
    assert res.json()["answer"] == "ok"


def test_documents_list(client, monkeypatch):
    monkeypatch.setattr(
        main, "list_documents",
        lambda: [{"filename": "a.pdf", "subject": "X", "topic": "Autres", "chunks": 3}],
    )
    res = client.get("/api/documents")
    assert res.status_code == 200
    assert res.json()["documents"][0]["filename"] == "a.pdf"


def test_delete_missing_document_404(client, monkeypatch):
    monkeypatch.setattr(main, "delete_document", lambda filename, subject=None: 0)
    res = client.request("DELETE", "/api/documents", json={"filename": "nope.pdf"})
    assert res.status_code == 404


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")
    monkeypatch.setattr(main, "delete_document", lambda filename, subject=None: 5)
    # Missing key -> 401
    res = client.request("DELETE", "/api/documents", json={"filename": "a.pdf"})
    assert res.status_code == 401
    # Correct key -> allowed
    res = client.request(
        "DELETE", "/api/documents",
        json={"filename": "a.pdf"}, headers={"X-API-Key": "secret"},
    )
    assert res.status_code == 200
