from pydantic import BaseModel, ConfigDict

from backend.schemas.llm import LLMResponse
from backend.services.metrics import ExecutionMetrics


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: int

    response: LLMResponse

    metrics: ExecutionMetrics

    metrics_warning: str | None = None
