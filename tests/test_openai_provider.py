import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.providers.exceptions import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
)
from backend.providers.openai_provider import OpenAIProvider
from backend.schemas.llm import LLMRequest


@pytest.fixture
def mock_client():
    client = MagicMock()

    client.with_options.return_value.responses.create = AsyncMock()

    return client


@pytest.fixture
def mock_response():
    return SimpleNamespace(
        id="resp_test_123",
        status="completed",
        output_text="Hello from BrainRoute!",
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=250,
            input_tokens_details=SimpleNamespace(
                cached_tokens=20,
            ),
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=80,
            ),
        ),
    )


def make_request(
    effort="medium",
    max_output_tokens=2048,
):
    return LLMRequest(
        prompt="Say hello",
        model_id="gpt-6-luna",
        effort=effort,
        max_output_tokens=max_output_tokens,
    )


def test_provider_name(mock_client):
    provider = OpenAIProvider(client=mock_client)

    assert provider.name == "openai"


def test_successful_execution(mock_client, mock_response):
    mock_client.with_options.return_value.responses.create.return_value = mock_response

    provider = OpenAIProvider(client=mock_client)

    response = asyncio.run(provider.execute(make_request()))

    assert response.provider == "openai"
    assert response.model_id == "gpt-6-luna"
    assert response.effort == "medium"

    assert response.status == "completed"
    assert response.output_text == "Hello from BrainRoute!"

    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 250
    assert response.usage.cached_input_tokens == 20
    assert response.usage.reasoning_tokens == 80

    assert response.latency_ms >= 0
    assert response.provider_response_id == "resp_test_123"


def test_correct_api_parameters(mock_client, mock_response):
    mock_client.with_options.return_value.responses.create.return_value = mock_response

    provider = OpenAIProvider(client=mock_client)

    asyncio.run(provider.execute(make_request(effort="high")))

    mock_client.with_options.assert_called_once_with(
        timeout=120.0,
        max_retries=0,
    )

    api_call = mock_client.with_options.return_value.responses.create

    kwargs = api_call.call_args.kwargs

    assert kwargs["model"] == "gpt-6-luna"
    assert kwargs["input"] == "Say hello"
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["max_output_tokens"] == 2048
    assert kwargs["store"] is False


def test_invalid_effort(mock_client):
    provider = OpenAIProvider(client=mock_client)

    with pytest.raises(ProviderConfigurationError):
        asyncio.run(provider.execute(make_request(effort="none")))

    mock_client.with_options.assert_not_called()


def test_unknown_model(mock_client):
    provider = OpenAIProvider(client=mock_client)

    request = LLMRequest(
        prompt="Hello",
        model_id="unknown-model",
        effort="medium",
    )

    with pytest.raises(ProviderConfigurationError):
        asyncio.run(provider.execute(request))

    mock_client.with_options.assert_not_called()


def test_incomplete_response(mock_client, mock_response):
    mock_response.status = "incomplete"
    mock_response.output_text = ""

    mock_response.incomplete_details = SimpleNamespace(reason="max_output_tokens")

    mock_client.with_options.return_value.responses.create.return_value = mock_response

    provider = OpenAIProvider(client=mock_client)

    response = asyncio.run(provider.execute(make_request()))

    assert response.status == "incomplete"
    assert response.finish_reason == "max_output_tokens"

    # Usage must still be preserved.
    assert response.usage.output_tokens == 250


def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(
        "backend.providers.openai_provider.get_settings",
        lambda: SimpleNamespace(openai_api_key=None),
    )

    with pytest.raises(ProviderAuthenticationError):
        OpenAIProvider()
