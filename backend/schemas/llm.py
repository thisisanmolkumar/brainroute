from pydantic import BaseModel, ConfigDict, Field

from backend.core.model_registry import ReasoningEffort


class LLMRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    prompt: str = Field(min_length=1)

    model_id: str = Field(min_length=1)

    effort: ReasoningEffort

    max_output_tokens: int | None = Field(
        default=None,
        gt=0,
    )

    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
    )


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)

    output_tokens: int = Field(ge=0)

    cached_input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
    )


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str

    model_id: str

    effort: ReasoningEffort

    output_text: str

    usage: TokenUsage | None = None

    latency_ms: float = Field(ge=0)

    provider_response_id: str | None = None

    finish_reason: str | None = None
