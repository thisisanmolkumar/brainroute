import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.v1.router import (
    get_execution_service,
    router,
)
from backend.routing.optimizer import optimize_route
from backend.routing.runtime import get_routing_service
from backend.routing.schemas import (
    CandidateEstimate,
)


class FakeRoutingService:
    def __init__(self):
        self.calls = 0

    def route(self, prompt, configs, policy):
        self.calls += 1

        estimates = [
            CandidateEstimate(
                model_id=config.model_id,
                effort=config.effort,
                success_probability=0.90,
                predicted_cost_usd="0.0004",
                predicted_latency_ms=1200,
            )
            for config in configs
        ]

        return optimize_route(
            estimates,
            policy,
            readiness_blockers=["Independent calibration unavailable"],
        )


class FakeExecutionService:
    def __init__(self):
        self.calls = []

    async def execute(self, request):
        self.calls.append(request)

        return {
            "execution_id": 123,
            "status": "completed",
            "output_text": "42",
        }


@pytest.fixture
def api_setup(monkeypatch):
    monkeypatch.delenv(
        "BRAINROUTE_ENABLE_LIVE_EXECUTION",
        raising=False,
    )

    monkeypatch.delenv(
        "BRAINROUTE_ROUTER_EXECUTION_TOKEN",
        raising=False,
    )

    app = FastAPI()

    app.include_router(
        router,
        prefix="/api/v1",
    )

    routing_service = FakeRoutingService()
    execution_service = FakeExecutionService()

    app.dependency_overrides[get_routing_service] = lambda: routing_service

    app.dependency_overrides[get_execution_service] = lambda: execution_service

    with TestClient(app) as client:
        yield (
            client,
            routing_service,
            execution_service,
        )


@pytest.fixture
def recommendation_payload():
    return {
        "prompt": "What is 17 multiplied by 24?",
        "candidates": [
            {
                "model_id": "model-a",
                "effort": "medium",
            }
        ],
        "policy": {
            "objective": "cost",
            "min_success_probability": 0.8,
            "max_cost_usd": "0.01",
        },
    }


@pytest.fixture
def execution_payload(recommendation_payload):
    return {
        **recommendation_payload,
        "approved_model_id": "model-a",
        "approved_effort": "medium",
        "approval": "I_APPROVE_PAID_EXECUTION",
        "acknowledge_uncalibrated_predictions": True,
        "max_output_tokens": 256,
        "timeout_seconds": 30,
    }


def test_recommendation_does_not_execute(
    api_setup,
    recommendation_payload,
):
    client, routing_service, execution_service = api_setup

    response = client.post(
        "/api/v1/router/recommend",
        json=recommendation_payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "review_required"
    assert result["recommendation"]["model_id"] == ("model-a")

    assert result["auto_execution_allowed"] is False

    assert routing_service.calls == 1
    assert execution_service.calls == []


def test_invalid_recommendation_rejected(api_setup):
    client, _, execution_service = api_setup

    response = client.post(
        "/api/v1/router/recommend",
        json={
            "prompt": "",
            "candidates": [],
        },
    )

    assert response.status_code == 422
    assert execution_service.calls == []


def test_execution_disabled_by_default(
    api_setup,
    execution_payload,
):
    client, _, execution_service = api_setup

    response = client.post(
        "/api/v1/router/execute",
        json=execution_payload,
    )

    assert response.status_code == 403
    assert execution_service.calls == []


def test_execution_requires_token(
    api_setup,
    execution_payload,
    monkeypatch,
):
    client, _, execution_service = api_setup

    monkeypatch.setenv(
        "BRAINROUTE_ENABLE_LIVE_EXECUTION",
        "true",
    )

    monkeypatch.setenv(
        "BRAINROUTE_ROUTER_EXECUTION_TOKEN",
        "test-secret",
    )

    response = client.post(
        "/api/v1/router/execute",
        json=execution_payload,
    )

    assert response.status_code == 401
    assert execution_service.calls == []


def test_wrong_token_rejected(
    api_setup,
    execution_payload,
    monkeypatch,
):
    client, _, execution_service = api_setup

    monkeypatch.setenv(
        "BRAINROUTE_ENABLE_LIVE_EXECUTION",
        "true",
    )

    monkeypatch.setenv(
        "BRAINROUTE_ROUTER_EXECUTION_TOKEN",
        "correct-secret",
    )

    response = client.post(
        "/api/v1/router/execute",
        json=execution_payload,
        headers={"X-BrainRoute-Execution-Token": ("wrong-secret")},
    )

    assert response.status_code == 401
    assert execution_service.calls == []


def test_uncalibrated_execution_requires_acknowledgement(
    api_setup,
    execution_payload,
    monkeypatch,
):
    client, _, execution_service = api_setup

    monkeypatch.setenv(
        "BRAINROUTE_ENABLE_LIVE_EXECUTION",
        "true",
    )

    monkeypatch.setenv(
        "BRAINROUTE_ROUTER_EXECUTION_TOKEN",
        "test-secret",
    )

    payload = {
        **execution_payload,
        "acknowledge_uncalibrated_predictions": False,
    }

    response = client.post(
        "/api/v1/router/execute",
        json=payload,
        headers={"X-BrainRoute-Execution-Token": ("test-secret")},
    )

    assert response.status_code == 409
    assert execution_service.calls == []


def test_approved_configuration_must_match(
    api_setup,
    execution_payload,
    monkeypatch,
):
    client, _, execution_service = api_setup

    monkeypatch.setenv(
        "BRAINROUTE_ENABLE_LIVE_EXECUTION",
        "true",
    )

    monkeypatch.setenv(
        "BRAINROUTE_ROUTER_EXECUTION_TOKEN",
        "test-secret",
    )

    payload = {
        **execution_payload,
        "approved_effort": "high",
    }

    response = client.post(
        "/api/v1/router/execute",
        json=payload,
        headers={"X-BrainRoute-Execution-Token": ("test-secret")},
    )

    assert response.status_code == 409
    assert execution_service.calls == []


def test_approved_execution(
    api_setup,
    execution_payload,
    monkeypatch,
):
    client, _, execution_service = api_setup

    monkeypatch.setenv(
        "BRAINROUTE_ENABLE_LIVE_EXECUTION",
        "true",
    )

    monkeypatch.setenv(
        "BRAINROUTE_ROUTER_EXECUTION_TOKEN",
        "test-secret",
    )

    response = client.post(
        "/api/v1/router/execute",
        json=execution_payload,
        headers={"X-BrainRoute-Execution-Token": ("test-secret")},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["decision"]["status"] == ("review_required")

    assert result["execution"]["status"] == ("completed")

    assert result["predicted_cost_is_hard_cap"] is False
    assert result["automatic_retry_enabled"] is False

    assert len(execution_service.calls) == 1

    executed_request = execution_service.calls[0]

    assert executed_request.model_id == "model-a"
    assert executed_request.effort == "medium"
    assert executed_request.max_output_tokens == 256
