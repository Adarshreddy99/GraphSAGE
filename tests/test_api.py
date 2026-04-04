import pytest
from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_health_endpoint():
    """Verify that the /health endpoint returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "total_papers" in response.json()

def test_recommend_endpoint_basic():
    """Verify that a basic recommendation request works."""
    payload = {
        "query": "Deep learning and graph neural networks",
        "k": 5
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    assert len(response.json()) == 5
    assert "title" in response.json()[0]

def test_recommend_invalid_k():
    """Verify that invalid K values are rejected by Pydantic."""
    payload = {
        "query": "test",
        "k": 1000  # Max is 50
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Validation error)

def test_recommend_empty_query():
    """Verify that an empty query results in a validation error or empty result."""
    payload = {
        "query": "",
        "k": 5
    }
    response = client.post("/recommend", json=payload)
    # Depending on implementation, might be 422 (if required) or 200 with empty list
    assert response.status_code in [200, 422]
