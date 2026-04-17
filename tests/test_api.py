"""Basic tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "llm_model" in data
        assert "total_chunks" in data
        print(f"\n✓ Health OK — {data['total_chunks']} chunks indexed")


class TestChat:
    def test_chat_requires_question(self, client):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422  # Pydantic validation error

    def test_chat_short_question(self, client):
        resp = client.post("/chat", json={"question": "Hi"})
        # Either 422 (too short) or 422 (no docs) is acceptable
        assert resp.status_code in (200, 422)

    @pytest.mark.skipif(
        True, reason="Requires documents to be ingested and Ollama running"
    )
    def test_chat_medical_question(self, client):
        resp = client.post(
            "/chat",
            json={"question": "What are the symptoms of Type 2 Diabetes?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 10
        assert "sources" in data
        assert isinstance(data["grounded"], bool)


class TestSources:
    def test_sources_endpoint(self, client):
        resp = client.get("/sources?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_chunks" in data
        assert "sample" in data


class TestReset:
    @pytest.mark.skipif(True, reason="Destructive — skip in CI")
    def test_reset(self, client):
        resp = client.delete("/reset")
        assert resp.status_code == 200
        assert "cleared" in resp.json()["message"].lower()
