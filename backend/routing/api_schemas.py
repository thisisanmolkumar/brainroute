from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from backend.routing.schemas import (
    CandidateConfig,
    RoutingDecision,
    RoutingPolicy,
)


class RecommendRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=12000,
    )

    candidates: list[CandidateConfig] = Field(
        min_length=1,
        max_length=24,
    )

    policy: RoutingPolicy = Field(default_factory=RoutingPolicy)

    @model_validator(mode="after")
    def validate_candidates(self):
        identities = [(item.model_id, item.effort) for item in self.candidates]

        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate routing candidates")

        return self


class ManualExecuteRequest(RecommendRequest):
    approved_model_id: str
    approved_effort: str

    approval: Literal["I_APPROVE_PAID_EXECUTION"]

    acknowledge_uncalibrated_predictions: bool = False

    max_output_tokens: int = Field(
        default=1024,
        ge=1,
        le=2048,
    )

    timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=120,
    )

    @model_validator(mode="after")
    def validate_execution_policy(self):
        if self.policy.max_cost_usd is None:
            raise ValueError(
                "Manual execution requires an explicit maximum predicted cost"
            )

        return self


class ManualExecutionResponse(BaseModel):
    decision: RoutingDecision

    execution: dict[str, Any]

    predicted_cost_is_hard_cap: bool = False
    automatic_retry_enabled: bool = False
