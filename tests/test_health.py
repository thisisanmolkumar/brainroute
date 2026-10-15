from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "BrainRoute"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_database_health():
    response = client.get("/health/db")

    assert response.status_code == 200
    assert response.json()["database"] == "connected"
