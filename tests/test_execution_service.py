import asyncio

import pytest

from backend.providers.base import BaseLLMProvider
from backend.providers.exceptions import (
    ProviderConfigurationError,
    ProviderTimeoutError,
)
from backend.schemas.llm import LLMRequest, LLMResponse, TokenUsage
from backend.services.execution_service import ExecutionService


class FakeProvider(BaseLLMProvider):
    def __init__(
        self,
        status="completed",
        should_fail=False,
    ):
        self.status = status
        self.should_fail = should_fail

    @property
    def name(self):
        return "openai"

    async def execute(self, request):

        if self.should_fail:
            raise ProviderTimeoutError("Simulated timeout")

        return LLMResponse(
            provider=self.name,
            model_id=request.model_id,
            effort=request.effort,
            output_text="BrainRoute response",
            status=self.status,
            service_tier="default",
            latency_ms=150.0,
            provider_response_id="resp_fake_123",
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=200,
                cached_input_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=50,
            ),
        )


class FakeRepository:
    def __init__(self):
        self.created = None
        self.completed = None
        self.failed = None

    def create_pending(
        self,
        request,
        provider,
        pricing_snapshot,
        category=None,
    ):
        self.created = {
            "request": request,
            "provider": provider,
            "pricing_snapshot": pricing_snapshot,
            "category": category,
        }

        return 42

    def complete(
        self,
        execution_id,
        response,
        metrics,
        metrics_warning=None,
    ):
        self.completed = {
            "execution_id": execution_id,
            "response": response,
            "metrics": metrics,
            "metrics_warning": metrics_warning,
        }

    def fail(self, execution_id, error, elapsed_ms):
        self.failed = {
            "execution_id": execution_id,
            "error": error,
            "elapsed_ms": elapsed_ms,
        }


def make_request():
    return LLMRequest(
        prompt="Explain binary search",
        model_id="gpt-6-luna",
        effort="medium",
        max_output_tokens=1024,
    )


def test_successful_execution():

    repository = FakeRepository()

    service = ExecutionService(
        provider=FakeProvider(),
        repository=repository,
    )

    result = asyncio.run(service.execute(make_request(), category="coding"))

    assert result.execution_id == 42

    assert result.response.status == "completed"
    assert result.response.output_text == "BrainRoute response"

    assert repository.created is not None
    assert repository.completed is not None
    assert repository.failed is None

    assert repository.created["category"] == "coding"
    assert repository.created["provider"] == "openai"

    assert "rates" in repository.created["pricing_snapshot"]

    assert result.metrics.cost is not None
    assert result.metrics.cost.total_cost_usd > 0


def test_execution_failure():

    repository = FakeRepository()

    service = ExecutionService(
        provider=FakeProvider(should_fail=True),
        repository=repository,
    )

    with pytest.raises(ProviderTimeoutError):
        asyncio.run(service.execute(make_request()))

    assert repository.created is not None
    assert repository.completed is None
    assert repository.failed is not None

    assert repository.failed["execution_id"] == 42

    assert isinstance(
        repository.failed["error"],
        ProviderTimeoutError,
    )


def test_incomplete_response():

    repository = FakeRepository()

    service = ExecutionService(
        provider=FakeProvider(status="incomplete"),
        repository=repository,
    )

    result = asyncio.run(service.execute(make_request()))

    assert result.response.status == "incomplete"

    # The incomplete response still has billable usage.
    assert result.metrics.cost is not None

    assert repository.completed is not None

    assert repository.completed["response"].status == "incomplete"


def test_invalid_model_does_not_create_record():

    repository = FakeRepository()

    service = ExecutionService(
        provider=FakeProvider(),
        repository=repository,
    )

    request = LLMRequest(
        prompt="Hello",
        model_id="nonexistent-model",
        effort="medium",
    )

    with pytest.raises(ProviderConfigurationError, match="Unknown model"):
        asyncio.run(service.execute(request))

    assert repository.created is None


def test_cost_snapshot():

    repository = FakeRepository()

    service = ExecutionService(
        provider=FakeProvider(),
        repository=repository,
    )

    asyncio.run(service.execute(make_request()))

    snapshot = repository.created["pricing_snapshot"]

    assert snapshot["currency"] == "USD"

    assert "input_per_1m_tokens" in snapshot["rates"]
    assert "output_per_1m_tokens" in snapshot["rates"]
