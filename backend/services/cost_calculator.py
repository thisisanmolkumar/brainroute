from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from backend.core.model_registry import ModelRegistry
from backend.schemas.llm import TokenUsage

MILLION = Decimal(1000000)


class UnsupportedPricingError(ValueError):
    """Pricing configuration cannot represent this execution."""


class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    currency: str
    pricing_updated: str | None

    ordinary_input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int

    input_cost_usd: Decimal
    cached_input_cost_usd: Decimal
    cache_write_cost_usd: Decimal
    output_cost_usd: Decimal

    total_cost_usd: Decimal

    estimated: bool = True
    notes: list[str] = Field(default_factory=list)


class CostCalculator:
    def __init__(
        self,
        registry: ModelRegistry | None = None,
    ):
        self.registry = registry or ModelRegistry()

    def calculate(
        self,
        model_id: str,
        usage: TokenUsage,
    ) -> CostBreakdown:

        model = self.registry.get_model(model_id)
        pricing = model.pricing

        if self.registry.config.currency != "USD":
            raise UnsupportedPricingError("Only USD pricing is supported")

        # ---------------------------------
        # 1. Validate pricing scope
        # ---------------------------------

        max_short = pricing.short_context_max_input_tokens

        if max_short is not None and usage.input_tokens > max_short:
            raise UnsupportedPricingError(
                f"Long-context pricing is not implemented for '{model_id}'"
            )

        # ---------------------------------
        # 2. Extract token counts
        # ---------------------------------

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        cached_tokens = usage.cached_input_tokens
        write_tokens = usage.cache_write_tokens

        notes = []

        if cached_tokens is None:
            cached_tokens = 0
            notes.append(
                "Cached input tokens not reported; assumed zero for estimation."
            )

        if write_tokens is None:
            write_tokens = 0
            notes.append(
                "Cache-write tokens not reported; assumed zero for estimation."
            )

        if cached_tokens + write_tokens > input_tokens:
            raise ValueError(
                "Cached and cache-write tokens cannot exceed total input tokens"
            )

        if (
            usage.reasoning_tokens is not None
            and usage.reasoning_tokens > output_tokens
        ):
            raise ValueError("Reasoning tokens cannot exceed total output tokens")

        ordinary_tokens = input_tokens - cached_tokens - write_tokens

        # ---------------------------------
        # 3. Validate available rates
        # ---------------------------------

        if cached_tokens > 0 and pricing.cached_input_per_1m_tokens is None:
            raise UnsupportedPricingError("Cached-input pricing is unavailable")

        if write_tokens > 0 and pricing.cache_write_per_1m_tokens is None:
            raise UnsupportedPricingError("Cache-write pricing is unavailable")

        # ---------------------------------
        # 4. Calculate component costs
        # ---------------------------------

        input_cost = Decimal(ordinary_tokens) * pricing.input_per_1m_tokens / MILLION

        cached_cost = (
            Decimal(cached_tokens)
            * (pricing.cached_input_per_1m_tokens or Decimal(0))
            / MILLION
        )

        write_cost = (
            Decimal(write_tokens)
            * (pricing.cache_write_per_1m_tokens or Decimal(0))
            / MILLION
        )

        output_cost = Decimal(output_tokens) * pricing.output_per_1m_tokens / MILLION

        total_cost = input_cost + cached_cost + write_cost + output_cost

        # ---------------------------------
        # 5. Return cost breakdown
        # ---------------------------------

        return CostBreakdown(
            model_id=model_id,
            currency="USD",
            pricing_updated=self.registry.config.pricing_updated,
            ordinary_input_tokens=ordinary_tokens,
            cached_input_tokens=cached_tokens,
            cache_write_tokens=write_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            cached_input_cost_usd=cached_cost,
            cache_write_cost_usd=write_cost,
            output_cost_usd=output_cost,
            total_cost_usd=total_cost,
            estimated=True,
            notes=notes,
        )
