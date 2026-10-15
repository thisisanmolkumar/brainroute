import asyncio

import pytest
from pydantic import ValidationError

from backend.providers.base import BaseLLMProvider
from backend.schemas.llm import (
    LLMRequest,
    LLMResponse,
    TokenUsage,
)


class FakeProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "fake"

    async def execute(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        return LLMResponse(
            provider=self.name,
            model_id=request.model_id,
            effort=request.effort,
            output_text=f"Echo: {request.prompt}",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=20,
            ),
            latency_ms=1.5,
            provider_response_id="fake-123",
        )


def test_provider_name():
    provider = FakeProvider()

    assert provider.name == "fake"


def test_provider_execution():
    provider = FakeProvider()

    request = LLMRequest(
        prompt="Explain binary search",
        model_id="test-model",
        effort="medium",
    )

    response = asyncio.run(provider.execute(request))

    assert isinstance(response, LLMResponse)

    assert response.provider == "fake"
    assert response.model_id == "test-model"
    assert response.effort == "medium"

    assert response.output_text == ("Echo: Explain binary search")

    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20


def test_invalid_effort():
    with pytest.raises(ValidationError):
        LLMRequest(
            prompt="Hello",
            model_id="test-model",
            effort="superhigh",
        )


def test_empty_prompt():
    with pytest.raises(ValidationError):
        LLMRequest(
            prompt="",
            model_id="test-model",
            effort="medium",
        )


def test_invalid_token_limit():
    with pytest.raises(ValidationError):
        LLMRequest(
            prompt="Hello",
            model_id="test-model",
            effort="medium",
            max_output_tokens=-100,
        )


def test_invalid_token_usage():
    with pytest.raises(ValidationError):
        TokenUsage(
            input_tokens=-10,
            output_tokens=20,
        )


def test_provider_interface_enforcement():

    class IncompleteProvider(BaseLLMProvider):
        pass

    with pytest.raises(TypeError):
        IncompleteProvider()
