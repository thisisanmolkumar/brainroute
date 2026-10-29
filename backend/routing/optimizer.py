import math
from decimal import Decimal

from backend.routing.schemas import (
    CandidateAssessment,
    CandidateEstimate,
    RoutingDecision,
    RoutingPolicy,
)


def candidate_name(
    candidate: CandidateEstimate,
) -> str:
    return f"{candidate.model_id} / {candidate.effort}"


def rejection_reasons(
    candidate: CandidateEstimate,
    policy: RoutingPolicy,
) -> list[str]:
    reasons = []

    if candidate.success_probability < policy.min_success_probability:
        reasons.append("Predicted success below minimum")

    needs_cost = policy.max_cost_usd is not None or policy.objective in {
        "cost",
        "balanced",
    }

    if needs_cost:
        if candidate.predicted_cost_usd is None:
            reasons.append("Cost prediction unavailable")

        elif (
            policy.max_cost_usd is not None
            and candidate.predicted_cost_usd > policy.max_cost_usd
        ):
            reasons.append("Predicted cost exceeds limit")

    needs_latency = policy.max_latency_ms is not None or policy.objective in {
        "latency",
        "balanced",
    }

    if needs_latency:
        if candidate.predicted_latency_ms is None:
            reasons.append("Latency prediction unavailable")

        elif (
            policy.max_latency_ms is not None
            and candidate.predicted_latency_ms > policy.max_latency_ms
        ):
            reasons.append("Predicted latency exceeds limit")

    return reasons


def objective_score(
    candidate: CandidateEstimate,
    policy: RoutingPolicy,
) -> float:

    if policy.objective == "quality":
        return 1.0 - candidate.success_probability

    if policy.objective == "cost":
        if candidate.predicted_cost_usd is None:
            raise ValueError("Cost prediction required")

        return float(candidate.predicted_cost_usd)

    if policy.objective == "latency":
        if candidate.predicted_latency_ms is None:
            raise ValueError("Latency prediction required")

        return candidate.predicted_latency_ms

    if (
        candidate.predicted_cost_usd is None
        or candidate.predicted_latency_ms is None
        or policy.cost_reference_usd is None
        or policy.latency_reference_ms is None
    ):
        raise ValueError(
            "Balanced routing needs complete predictions and reference values"
        )

    success_penalty = 1.0 - candidate.success_probability

    cost_penalty = float(candidate.predicted_cost_usd / policy.cost_reference_usd)

    latency_penalty = candidate.predicted_latency_ms / policy.latency_reference_ms

    return (
        policy.success_weight * success_penalty
        + policy.cost_weight * cost_penalty
        + policy.latency_weight * latency_penalty
    )


def ranking_key(
    assessment: CandidateAssessment,
    policy: RoutingPolicy,
) -> tuple:

    candidate = assessment.estimate

    cost = (
        candidate.predicted_cost_usd
        if candidate.predicted_cost_usd is not None
        else Decimal("Infinity")
    )

    latency = (
        candidate.predicted_latency_ms
        if candidate.predicted_latency_ms is not None
        else math.inf
    )

    probability = candidate.success_probability

    if policy.objective == "quality":
        return (
            -probability,
            cost,
            latency,
            candidate.model_id,
            candidate.effort,
        )

    if policy.objective == "cost":
        return (
            cost,
            -probability,
            latency,
            candidate.model_id,
            candidate.effort,
        )

    if policy.objective == "latency":
        return (
            latency,
            -probability,
            cost,
            candidate.model_id,
            candidate.effort,
        )

    return (
        assessment.objective_score,
        -probability,
        cost,
        latency,
        candidate.model_id,
        candidate.effort,
    )


def dominates(
    better: CandidateEstimate,
    worse: CandidateEstimate,
) -> bool:

    if (
        better.predicted_cost_usd is None
        or worse.predicted_cost_usd is None
        or better.predicted_latency_ms is None
        or worse.predicted_latency_ms is None
    ):
        return False

    no_worse = (
        better.success_probability >= worse.success_probability
        and better.predicted_cost_usd <= worse.predicted_cost_usd
        and better.predicted_latency_ms <= worse.predicted_latency_ms
    )

    strictly_better = (
        better.success_probability > worse.success_probability
        or better.predicted_cost_usd < worse.predicted_cost_usd
        or better.predicted_latency_ms < worse.predicted_latency_ms
    )

    return no_worse and strictly_better


def optimize_route(
    estimates: list[CandidateEstimate],
    policy: RoutingPolicy,
    *,
    readiness_blockers: list[str] | None = None,
    warnings: list[str] | None = None,
) -> RoutingDecision:

    if not estimates:
        raise ValueError("No candidate predictions supplied")

    blockers = list(readiness_blockers or [])
    notes = list(warnings or [])

    identities = [(item.model_id, item.effort) for item in estimates]

    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate model/effort candidates")

    # ----------------------------------
    # 1. Check constraints
    # ----------------------------------

    assessments = []

    for estimate in estimates:
        reasons = rejection_reasons(
            estimate,
            policy,
        )

        assessment = CandidateAssessment(
            estimate=estimate,
            eligible=not reasons,
            rejection_reasons=reasons,
        )

        if assessment.eligible:
            assessment.objective_score = objective_score(estimate, policy)

        assessments.append(assessment)

    eligible = [assessment for assessment in assessments if assessment.eligible]

    # ----------------------------------
    # 2. No eligible configurations
    # ----------------------------------

    if not eligible:
        return RoutingDecision(
            status="no_eligible_candidates",
            objective=policy.objective,
            recommendation=None,
            assessments=assessments,
            readiness_blockers=blockers,
            warnings=notes,
        )

    # ----------------------------------
    # 3. Pareto analysis
    # ----------------------------------

    pareto_frontier = []

    complete_metrics = all(
        item.estimate.predicted_cost_usd is not None
        and item.estimate.predicted_latency_ms is not None
        for item in eligible
    )

    if complete_metrics:
        for candidate in eligible:
            candidate.dominated = any(
                dominates(
                    other.estimate,
                    candidate.estimate,
                )
                for other in eligible
                if other is not candidate
            )

        pareto_frontier = sorted(
            candidate_name(item.estimate) for item in eligible if not item.dominated
        )

    else:
        notes.append(
            "Pareto analysis omitted because some eligible "
            "candidates lack cost or latency predictions."
        )

    # ----------------------------------
    # 4. Rank eligible configurations
    # ----------------------------------

    ranked = sorted(
        eligible,
        key=lambda assessment: ranking_key(
            assessment,
            policy,
        ),
    )

    for rank, assessment in enumerate(
        ranked,
        start=1,
    ):
        assessment.rank = rank

    recommendation = ranked[0].estimate

    # ----------------------------------
    # 5. Return advisory recommendation
    # ----------------------------------

    status = "review_required" if blockers else "provisional"

    return RoutingDecision(
        status=status,
        objective=policy.objective,
        recommendation=recommendation,
        assessments=assessments,
        pareto_frontier=pareto_frontier,
        readiness_blockers=blockers,
        warnings=notes,
        auto_execution_allowed=False,
    )
