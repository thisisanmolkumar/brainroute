from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.llm import LLMRequest

# ----------------------------------------
# Execution Request
# ----------------------------------------


class ExecuteRequest(LLMRequest):
    prompt: str = Field(
        min_length=1,
        max_length=20_000,
    )

    max_output_tokens: int = Field(
        default=1024,
        ge=16,
        le=4096,
    )

    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=120,
    )

    category: str | None = Field(
        default=None,
        max_length=64,
    )

    max_cost_usd: Decimal | None = Field(
        default=None,
        gt=0,
    )


# ----------------------------------------
# Execution History
# ----------------------------------------


class ExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    task_id: int

    provider: str
    model_id: str
    effort: str

    status: str
    success: bool | None

    prompt_preview: str

    input_tokens: int | None
    output_tokens: int | None

    cost_usd: Decimal | None

    latency_ms: int | None
    end_to_end_latency_ms: int | None

    created_at: datetime
    finished_at: datetime | None


class ExecutionDetail(ExecutionSummary):
    prompt: str
    category: str | None

    output_text: str | None

    provider_response_id: str | None
    service_tier: str | None

    cached_input_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None

    pricing_snapshot: dict | None
    metrics_warning: str | None

    error_type: str | None
    error_message: str | None


class ExecutionHistoryPage(BaseModel):
    total: int
    limit: int
    offset: int

    items: list[ExecutionSummary]
