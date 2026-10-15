from time import perf_counter

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)

from backend.core.config import get_settings
from backend.core.model_registry import ModelRegistry
from backend.providers.base import BaseLLMProvider
from backend.providers.exceptions import (
    LLMProviderError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from backend.schemas.llm import LLMRequest, LLMResponse, TokenUsage


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        registry: ModelRegistry | None = None,
        client: AsyncOpenAI | None = None,
    ):
        self.registry = registry or ModelRegistry()

        self._owns_client = client is None

        if client is not None:
            self._client = client
        else:
            api_key = get_settings().openai_api_key

            if not api_key or not api_key.strip():
                raise ProviderAuthenticationError("OPENAI_API_KEY is not configured")

            self._client = AsyncOpenAI(
                api_key=api_key,
                max_retries=0,
            )

    @property
    def name(self) -> str:
        return "openai"

    async def execute(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        # ----------------------------------
        # 1. Validate model configuration
        # ----------------------------------

        try:
            model = self.registry.validate_configuration(
                model_id=request.model_id,
                effort=request.effort,
            )

        except ValueError as exc:
            raise ProviderConfigurationError(str(exc)) from exc

        if model.provider != self.name:
            raise ProviderConfigurationError(
                f"Model '{request.model_id}' belongs to provider '{model.provider}'"
            )

        # ----------------------------------
        # 2. Prepare API parameters
        # ----------------------------------

        max_tokens = (
            request.max_output_tokens
            if request.max_output_tokens is not None
            else self.registry.config.defaults.max_output_tokens
        )

        # ----------------------------------
        # 3. Execute request
        # ----------------------------------

        start_time = perf_counter()

        try:
            response = await self._client.with_options(
                timeout=request.timeout_seconds,
                max_retries=0,
            ).responses.create(
                model=request.model_id,
                input=request.prompt,
                reasoning={
                    "effort": request.effort,
                },
                max_output_tokens=max_tokens,
                store=False,
            )

        except (AuthenticationError, PermissionDeniedError) as exc:
            raise ProviderAuthenticationError(
                "OpenAI rejected the API credentials or permissions"
            ) from exc

        except RateLimitError as exc:
            raise ProviderRateLimitError("OpenAI rate limit exceeded") from exc

        except APITimeoutError as exc:
            raise ProviderTimeoutError("OpenAI request timed out") from exc

        except BadRequestError as exc:
            raise ProviderConfigurationError(
                "OpenAI rejected the request parameters"
            ) from exc

        except APIConnectionError as exc:
            raise LLMProviderError("Failed to connect to OpenAI") from exc

        except APIStatusError as exc:
            if exc.status_code == 404:
                raise ProviderConfigurationError(
                    f"Model '{request.model_id}' was not found "
                    "or is unavailable to this API project"
                ) from exc

            raise LLMProviderError(f"OpenAI returned HTTP {exc.status_code}") from exc

        except APIError as exc:
            raise LLMProviderError("Unexpected OpenAI API error") from exc

        # ----------------------------------
        # 4. Calculate latency
        # ----------------------------------

        latency_ms = (perf_counter() - start_time) * 1000

        # ----------------------------------
        # 5. Extract token usage
        # ----------------------------------

        token_usage = None

        if response.usage is not None:
            usage = response.usage

            input_details = getattr(
                usage,
                "input_tokens_details",
                None,
            )

            output_details = getattr(
                usage,
                "output_tokens_details",
                None,
            )

            token_usage = TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=getattr(
                    input_details,
                    "cached_tokens",
                    None,
                ),
                reasoning_tokens=getattr(
                    output_details,
                    "reasoning_tokens",
                    None,
                ),
            )

        # ----------------------------------
        # 6. Extract response information
        # ----------------------------------

        status = response.status or "unknown"

        incomplete_details = getattr(
            response,
            "incomplete_details",
            None,
        )

        finish_reason = getattr(
            incomplete_details,
            "reason",
            None,
        )

        if status in {"failed", "cancelled"}:
            # A response was returned, but it did not
            # complete successfully. Preserve its data
            # for the execution-tracking layer.
            pass

        if status == "completed" and response.output_text is None:
            raise ProviderResponseError(
                "OpenAI completed without a usable text response"
            )

        # ----------------------------------
        # 7. Return standardized response
        # ----------------------------------

        return LLMResponse(
            provider=self.name,
            model_id=request.model_id,
            effort=request.effort,
            output_text=response.output_text or "",
            status=status,
            usage=token_usage,
            latency_ms=latency_ms,
            provider_response_id=response.id,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()
