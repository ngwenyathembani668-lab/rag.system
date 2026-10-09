import os
import sys

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

dotenv_path = os.path.join(project_root, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient


def test_health_report_is_ok_even_when_rag_is_unconfigured():
    """The app should still start up cleanly and expose the real configuration issue."""
    from app.main import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post("/ask", json={"question": "when does the course start?"})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "Missing Gemini API key" in detail or "RAG pipeline is not initialized" in detail
