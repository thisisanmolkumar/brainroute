import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from backend.benchmarks.ml_dataset import (
    MLExample,
    MLMetadata,
    MLTargets,
    TargetMasks,
    extract_features,
)
from backend.routing.optimizer import optimize_route
from backend.routing.schemas import (
    CandidateConfig,
    CandidateEstimate,
    RoutingPolicy,
)
from backend.routing.service import RoutingService


def sample_estimates():
    return [
        CandidateEstimate(
            model_id="luna",
            effort="medium",
            success_probability=0.72,
            predicted_cost_usd=Decimal("0.0003"),
            predicted_latency_ms=1500,
        ),
        CandidateEstimate(
            model_id="luna",
            effort="high",
            success_probability=0.86,
            predicted_cost_usd=Decimal("0.0007"),
            predicted_latency_ms=2800,
        ),
        CandidateEstimate(
            model_id="mini",
            effort="medium",
            success_probability=0.79,
            predicted_cost_usd=Decimal("0.0004"),
            predicted_latency_ms=1900,
        ),
        CandidateEstimate(
            model_id="mini",
            effort="high",
            success_probability=0.93,
            predicted_cost_usd=Decimal("0.0012"),
            predicted_latency_ms=3400,
        ),
    ]


def test_cost_optimized_routing():
    decision = optimize_route(
        sample_estimates(),
        RoutingPolicy(
            objective="cost",
            min_success_probability=0.80,
            max_cost_usd=Decimal("0.001"),
        ),
    )

    assert decision.status == "provisional"

    assert decision.recommendation.model_id == "luna"
    assert decision.recommendation.effort == "high"

    assert decision.auto_execution_allowed is False


def test_latency_optimized_routing():
    decision = optimize_route(
        sample_estimates(),
        RoutingPolicy(
            objective="latency",
            min_success_probability=0.70,
        ),
    )

    assert decision.recommendation.model_id == "luna"
    assert decision.recommendation.effort == "medium"


def test_quality_optimized_routing():
    decision = optimize_route(
        sample_estimates(),
        RoutingPolicy(
            objective="quality",
            min_success_probability=0.80,
        ),
    )

    assert decision.recommendation.model_id == "mini"
    assert decision.recommendation.effort == "high"


def test_balanced_routing():
    decision = optimize_route(
        sample_estimates(),
        RoutingPolicy(
            objective="balanced",
            min_success_probability=0.80,
            cost_reference_usd=Decimal("0.002"),
            latency_reference_ms=5000,
        ),
    )

    assert decision.recommendation.model_id == "luna"
    assert decision.recommendation.effort == "high"


def test_no_eligible_candidate():
    decision = optimize_route(
        sample_estimates(),
        RoutingPolicy(
            objective="cost",
            min_success_probability=0.99,
        ),
    )

    assert decision.status == "no_eligible_candidates"
    assert decision.recommendation is None


def test_missing_cost_is_not_zero():
    estimates = [
        CandidateEstimate(
            model_id="model-a",
            effort="medium",
            success_probability=0.95,
            predicted_cost_usd=None,
            predicted_latency_ms=1000,
        ),
    ]

    decision = optimize_route(
        estimates,
        RoutingPolicy(
            objective="cost",
            min_success_probability=0.80,
        ),
    )

    assert decision.recommendation is None
    assert "Cost prediction unavailable" in (decision.assessments[0].rejection_reasons)


def test_quality_allows_missing_cost():
    estimates = [
        CandidateEstimate(
            model_id="model-a",
            effort="medium",
            success_probability=0.95,
            predicted_cost_usd=None,
            predicted_latency_ms=1000,
        ),
    ]

    decision = optimize_route(
        estimates,
        RoutingPolicy(
            objective="quality",
            min_success_probability=0.80,
        ),
    )

    assert decision.recommendation is not None
    assert decision.pareto_frontier == []


def test_pareto_dominance():
    estimates = [
        CandidateEstimate(
            model_id="a",
            effort="high",
            success_probability=0.90,
            predicted_cost_usd=Decimal("0.001"),
            predicted_latency_ms=1000,
        ),
        CandidateEstimate(
            model_id="b",
            effort="high",
            success_probability=0.80,
            predicted_cost_usd=Decimal("0.002"),
            predicted_latency_ms=2000,
        ),
        CandidateEstimate(
            model_id="c",
            effort="high",
            success_probability=0.95,
            predicted_cost_usd=Decimal("0.003"),
            predicted_latency_ms=2500,
        ),
    ]

    decision = optimize_route(
        estimates,
        RoutingPolicy(
            objective="cost",
            min_success_probability=0,
        ),
    )

    assert set(decision.pareto_frontier) == {
        "a / high",
        "c / high",
    }

    assessments = {item.estimate.model_id: item for item in decision.assessments}

    assert assessments["b"].dominated is True
    assert assessments["a"].dominated is False


def test_duplicate_candidate_rejected():
    estimate = sample_estimates()[0]

    with pytest.raises(
        ValueError,
        match="Duplicate model/effort",
    ):
        optimize_route(
            [estimate, estimate],
            RoutingPolicy(),
        )


def test_readiness_blocks_automatic_trust():
    decision = optimize_route(
        sample_estimates(),
        RoutingPolicy(
            objective="quality",
        ),
        readiness_blockers=["Validation contains only successes"],
    )

    assert decision.status == "review_required"
    assert decision.recommendation is not None
    assert decision.auto_execution_allowed is False


def test_balanced_requires_references():
    with pytest.raises(
        ValueError,
        match="Balanced routing requires",
    ):
        RoutingPolicy(
            objective="balanced",
        )


class FakeRegistry:
    def validate_configuration(
        self,
        model_id,
        effort,
    ):
        if model_id != "model-a":
            raise ValueError("Unknown model")

        if effort not in {"medium", "high"}:
            raise ValueError("Unsupported effort")

        return SimpleNamespace(
            enabled=True,
            provider="openai",
        )


class FakePredictor:
    def __init__(self):
        self.preprocessing = {
            "models": ["model-a"],
            "efforts": ["medium", "high"],
        }

        self.calls = []

    def predict_many(self, requests):
        self.calls.append(requests)

        results = []

        for _, model_id, effort in requests:
            high = effort == "high"

            results.append(
                {
                    "model_id": model_id,
                    "effort": effort,
                    "success_probability": (0.95 if high else 0.85),
                    "predicted_cost_usd": (0.0005 if high else 0.0002),
                    "predicted_latency_ms": (2000 if high else 1000),
                }
            )

        return results


def make_service_example(
    task_index,
    effort,
    split,
    successes,
):
    trials = 2

    return MLExample(
        features=extract_features(
            prompt=f"Solve test problem {task_index}.",
            model_id="model-a",
            effort=effort,
        ),
        targets=MLTargets(
            success_rate=successes / trials,
            successes=successes,
            trials=trials,
            mean_cost_usd=Decimal("0.001"),
            mean_latency_ms=100,
            cost_observations=trials,
            latency_observations=trials,
        ),
        masks=TargetMasks(
            cost=True,
            latency=True,
        ),
        metadata=MLMetadata(
            source="synthetic",
            task_id=f"task_{task_index}",
            domain="math",
            family_id=None,
            original_split=split,
            group_key=(f"task:synthetic:task_{task_index}"),
            run_ids=[1],
        ),
    )


def write_service_examples(path, examples):
    path.write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n",
        encoding="utf-8",
    )


def build_service(
    tmp_path,
    *,
    all_validation_success=False,
):
    train = [
        make_service_example(
            index,
            effort,
            "train",
            successes=1 if index % 2 == 0 else 2,
        )
        for index in range(4)
        for effort in ["medium", "high"]
    ]

    validation = [
        make_service_example(
            index,
            effort,
            "validation",
            successes=(2 if all_validation_success else 1 if index % 2 == 0 else 2),
        )
        for index in range(4, 6)
        for effort in ["medium", "high"]
    ]

    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    report_path = tmp_path / "report.json"

    write_service_examples(train_path, train)
    write_service_examples(
        validation_path,
        validation,
    )

    report_path.write_text(
        json.dumps(
            {
                "model_type": "pytorch_multitask_router",
                "dataset": {
                    "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                    "validation_sha256": hashlib.sha256(
                        validation_path.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    predictor = FakePredictor()

    service = RoutingService(
        predictor=predictor,
        registry=FakeRegistry(),
        train_path=train_path,
        validation_path=validation_path,
        report_path=report_path,
        min_validation_groups=1,
        min_train_groups_per_config=1,
        min_outcomes_per_class=1,
    )

    return service, predictor, train_path


def test_service_batch_prediction(tmp_path):
    service, predictor, _ = build_service(tmp_path)

    decision = service.route(
        prompt="Solve a new mathematical problem.",
        configs=[
            CandidateConfig(
                model_id="model-a",
                effort="medium",
            ),
            CandidateConfig(
                model_id="model-a",
                effort="high",
            ),
        ],
        policy=RoutingPolicy(
            objective="quality",
            min_success_probability=0.8,
        ),
    )

    assert len(predictor.calls) == 1
    assert len(predictor.calls[0]) == 2

    assert decision.recommendation.effort == "high"
    assert decision.status == "provisional"
    assert decision.auto_execution_allowed is False


def test_service_flags_one_class_validation(tmp_path):
    service, _, _ = build_service(
        tmp_path,
        all_validation_success=True,
    )

    decision = service.route(
        prompt="Solve this problem.",
        configs=[
            CandidateConfig(
                model_id="model-a",
                effort="medium",
            ),
        ],
        policy=RoutingPolicy(
            objective="cost",
            min_success_probability=0.8,
        ),
    )

    assert decision.status == "review_required"

    assert any("one observed" in reason for reason in decision.readiness_blockers)


def test_service_rejects_untrained_configuration(
    tmp_path,
):
    service, _, _ = build_service(tmp_path)

    with pytest.raises(
        ValueError,
        match="Unsupported effort",
    ):
        service.route(
            prompt="Solve a problem.",
            configs=[
                CandidateConfig(
                    model_id="model-a",
                    effort="xhigh",
                ),
            ],
            policy=RoutingPolicy(),
        )


def test_service_rejects_dataset_changes(tmp_path):
    service, predictor, train_path = build_service(tmp_path)

    # Existing service has already loaded its data,
    # so construct a new service after modification.
    train_path.write_text(
        train_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="hashes do not match",
    ):
        RoutingService(
            predictor=predictor,
            registry=FakeRegistry(),
            train_path=train_path,
            validation_path=(tmp_path / "validation.jsonl"),
            report_path=tmp_path / "report.json",
        )

    assert service is not None
