import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a fresh FastAPI test client with startup events triggered."""
    with TestClient(app) as test_client:
        yield test_client


def test_ask_endpoint_valid_question(client):
    """Part 4: Test a real question expected to match handbook rules."""
    response = client.post("/ask", json={"question": "What are the hardware requirements for a laptop?"})
    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "source" in payload
    assert payload["answer"] != "I don't know."


def test_ask_endpoint_missing_info(client):
    """Part 4 & Part 2 Guardrail: Test out-of-bounds question returns 'I don't know.'."""
    response = client.post("/ask", json={"question": "How do I build a rocket ship in the school yard?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "I don't know."
    assert payload["source"] == "N/A"


def test_invalid_payload_handling(client):
    """Part 5: Verify that malformed JSON payloads handle errors gracefully (HTTP 422)."""
    response = client.post("/ask", json={})
    assert response.status_code == 422
