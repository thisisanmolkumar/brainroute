import pytest
from fastapi.testclient import TestClient

from backend.api.dependencies import (
    get_execution_service,
    get_repository,
)
from backend.main import app
from backend.providers.exceptions import (
    ProviderRateLimitError,
)
from backend.schemas.api import ExecutionHistoryPage
from backend.schemas.execution import ExecutionResult
from backend.schemas.llm import LLMResponse
from backend.services.metrics import ExecutionMetrics

# ----------------------------------------
# Fake Execution Service
# ----------------------------------------


class FakeExecutionService:
    def __init__(self):
        self.calls = []
        self.error = None

    async def execute(self, request, category=None):

        self.calls.append(
            {
                "request": request,
                "category": category,
            }
        )

        if self.error is not None:
            raise self.error

        response = LLMResponse(
            provider="openai",
            model_id=request.model_id,
            effort=request.effort,
            status="completed",
            output_text="BrainRoute test response",
            latency_ms=100,
        )

        metrics = ExecutionMetrics(
            provider="openai",
            model_id=request.model_id,
            effort=request.effort,
            status="completed",
            provider_latency_ms=100,
            end_to_end_latency_ms=110,
            overhead_ms=10,
            usage=None,
            cost=None,
        )

        return ExecutionResult(
            execution_id=42,
            response=response,
            metrics=metrics,
        )


# ----------------------------------------
# Fake Repository
# ----------------------------------------


class FakeRepository:
    def list_executions(self, limit=20, offset=0):
        return ExecutionHistoryPage(
            total=0,
            limit=limit,
            offset=offset,
            items=[],
        )

    def get_execution(self, execution_id):
        return None


# ----------------------------------------
# Test Fixture
# ----------------------------------------


@pytest.fixture
def test_client():

    service = FakeExecutionService()
    repository = FakeRepository()

    app.dependency_overrides[get_execution_service] = lambda: service

    app.dependency_overrides[get_repository] = lambda: repository

    try:
        with TestClient(app) as client:
            yield client, service

    finally:
        app.dependency_overrides.clear()


# ----------------------------------------
# Model Endpoint Tests
# ----------------------------------------


def test_list_models(test_client):
    client, _ = test_client

    response = client.get("/api/v1/models")

    assert response.status_code == 200
    assert isinstance(response.json(), list)

    ids = [model["id"] for model in response.json()]

    assert "gpt-6-luna" in ids


def test_get_model(test_client):
    client, _ = test_client

    response = client.get("/api/v1/models/gpt-6-luna")

    assert response.status_code == 200
    assert response.json()["id"] == "gpt-6-luna"


def test_unknown_model(test_client):
    client, _ = test_client

    response = client.get("/api/v1/models/nonexistent-model")

    assert response.status_code == 404


# ----------------------------------------
# Execution Endpoint Tests
# ----------------------------------------


def test_execute_success(test_client):
    client, service = test_client

    response = client.post(
        "/api/v1/execute",
        json={
            "prompt": "Explain binary search.",
            "model_id": "gpt-6-luna",
            "effort": "medium",
            "max_output_tokens": 1024,
            "category": "coding",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["execution_id"] == 42
    assert data["response"]["status"] == "completed"

    assert len(service.calls) == 1

    assert service.calls[0]["category"] == "coding"


def test_invalid_effort(test_client):
    client, service = test_client

    response = client.post(
        "/api/v1/execute",
        json={
            "prompt": "Hello",
            "model_id": "gpt-6-luna",
            "effort": "invalid",
        },
    )

    assert response.status_code == 422
    assert len(service.calls) == 0


def test_invalid_model(test_client):
    client, service = test_client

    response = client.post(
        "/api/v1/execute",
        json={
            "prompt": "Hello",
            "model_id": "nonexistent-model",
            "effort": "medium",
        },
    )

    assert response.status_code == 400
    assert len(service.calls) == 0


def test_output_token_limit(test_client):
    client, service = test_client

    response = client.post(
        "/api/v1/execute",
        json={
            "prompt": "Hello",
            "model_id": "gpt-6-luna",
            "effort": "medium",
            "max_output_tokens": 50000,
        },
    )

    assert response.status_code == 422
    assert len(service.calls) == 0


def test_budget_rejection(test_client):
    client, service = test_client

    response = client.post(
        "/api/v1/execute",
        json={
            "prompt": "Explain binary search.",
            "model_id": "gpt-6-luna",
            "effort": "medium",
            "max_output_tokens": 1024,
            "max_cost_usd": "0.000000001",
        },
    )

    assert response.status_code == 400
    assert len(service.calls) == 0


def test_provider_rate_limit(test_client):
    client, service = test_client

    service.error = ProviderRateLimitError("Simulated rate limit")

    response = client.post(
        "/api/v1/execute",
        json={
            "prompt": "Hello",
            "model_id": "gpt-6-luna",
            "effort": "medium",
        },
    )

    assert response.status_code == 429


# ----------------------------------------
# Execution History Tests
# ----------------------------------------


def test_execution_history(test_client):
    client, _ = test_client

    response = client.get("/api/v1/executions?limit=10&offset=0")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 0
    assert data["limit"] == 10
    assert data["offset"] == 0
    assert data["items"] == []


def test_invalid_pagination(test_client):
    client, _ = test_client

    response = client.get("/api/v1/executions?limit=1000")

    assert response.status_code == 422


def test_execution_not_found(test_client):
    client, _ = test_client

    response = client.get("/api/v1/executions/99999")

    assert response.status_code == 404
