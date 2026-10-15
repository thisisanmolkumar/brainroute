from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.llm import LLMResponse, TokenUsage
from backend.services.cost_calculator import (
    CostBreakdown,
    CostCalculator,
    UnsupportedPricingError,
)


class ExecutionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str
    effort: str
    status: str

    provider_latency_ms: float = Field(ge=0)
    end_to_end_latency_ms: float = Field(ge=0)
    overhead_ms: float = Field(ge=0)

    usage: TokenUsage | None = None
    cost: CostBreakdown | None = None


class MetricsCollector:
    def __init__(
        self,
        calculator: CostCalculator | None = None,
    ):
        self.calculator = calculator or CostCalculator()

    def collect(
        self,
        response: LLMResponse,
        end_to_end_latency_ms: float,
    ) -> ExecutionMetrics:

        if end_to_end_latency_ms < 0:
            raise ValueError("End-to-end latency cannot be negative")

        # ---------------------------------
        # 1. Calculate cost
        # ---------------------------------

        cost = None

        if response.usage is not None:
            if response.service_tier not in (
                None,
                "default",
            ):
                raise UnsupportedPricingError(
                    "Cost calculation currently supports only standard processing"
                )

            model = self.calculator.registry.get_model(response.model_id)

            if model.provider != response.provider:
                raise ValueError(
                    "Response provider does not match the registered model provider"
                )

            cost = self.calculator.calculate(
                model_id=response.model_id,
                usage=response.usage,
            )

        # ---------------------------------
        # 2. Calculate latency
        # ---------------------------------

        provider_latency = response.latency_ms

        overhead = max(
            0.0,
            end_to_end_latency_ms - provider_latency,
        )

        # ---------------------------------
        # 3. Return execution metrics
        # ---------------------------------

        return ExecutionMetrics(
            provider=response.provider,
            model_id=response.model_id,
            effort=response.effort,
            status=response.status,
            provider_latency_ms=provider_latency,
            end_to_end_latency_ms=end_to_end_latency_ms,
            overhead_ms=overhead,
            usage=response.usage,
            cost=cost,
        )
