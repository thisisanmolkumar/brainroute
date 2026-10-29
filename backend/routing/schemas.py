from decimal import Decimal
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

RoutingObjective = Literal[
    "cost",
    "latency",
    "quality",
    "balanced",
]


class CandidateConfig(BaseModel):
    model_id: str = Field(min_length=1)
    effort: str = Field(min_length=1)


class CandidateEstimate(CandidateConfig):
    success_probability: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    predicted_cost_usd: Decimal | None = Field(
        default=None,
        ge=0,
    )

    predicted_latency_ms: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )


class RoutingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: RoutingObjective = "cost"

    min_success_probability: float = Field(
        default=0.80,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    max_cost_usd: Decimal | None = Field(
        default=None,
        gt=0,
    )

    max_latency_ms: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    # Used only for balanced optimization.
    success_weight: float = Field(
        default=0.60,
        ge=0,
        allow_inf_nan=False,
    )

    cost_weight: float = Field(
        default=0.25,
        ge=0,
        allow_inf_nan=False,
    )

    latency_weight: float = Field(
        default=0.15,
        ge=0,
        allow_inf_nan=False,
    )

    cost_reference_usd: Decimal | None = Field(
        default=None,
        gt=0,
    )

    latency_reference_ms: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_balanced_policy(self):
        if self.objective != "balanced":
            return self

        if self.cost_reference_usd is None or self.latency_reference_ms is None:
            raise ValueError(
                "Balanced routing requires cost_reference_usd and latency_reference_ms"
            )

        weight_sum = self.success_weight + self.cost_weight + self.latency_weight

        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("Balanced routing weights must sum to 1")

        return self


class CandidateAssessment(BaseModel):
    estimate: CandidateEstimate

    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)

    dominated: bool = False
    rank: int | None = None
    objective_score: float | None = None


class RoutingDecision(BaseModel):
    status: Literal[
        "provisional",
        "review_required",
        "no_eligible_candidates",
    ]

    objective: RoutingObjective

    recommendation: CandidateEstimate | None

    assessments: list[CandidateAssessment]

    pareto_frontier: list[str] = Field(default_factory=list)

    readiness_blockers: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)

    auto_execution_allowed: bool = False
