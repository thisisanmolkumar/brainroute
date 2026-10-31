import hashlib
import json
from decimal import Decimal

import pytest

from backend.benchmarks.ml_dataset import (
    MLExample,
    MLMetadata,
    MLTargets,
    TargetMasks,
    extract_features,
)
from backend.evaluation.calibration import fit_temperature
from backend.evaluation.metrics import (
    pairwise_ranking_report,
    probability_report,
)
from backend.evaluation.replay import replay_policies
from backend.evaluation.runner import evaluate_router
from backend.routing.schemas import (
    CandidateConfig,
    RoutingPolicy,
)

CONFIGS = [
    CandidateConfig(model_id="a", effort="medium"),
    CandidateConfig(model_id="b", effort="medium"),
]


def make_example(index, model_id, split):
    trials = 2

    successes = 1 if model_id == "a" else 2

    return MLExample(
        features=extract_features(
            prompt=f"Evaluate synthetic question {index}.",
            model_id=model_id,
            effort="medium",
        ),
        targets=MLTargets(
            successes=successes,
            trials=trials,
            success_rate=successes / trials,
            mean_cost_usd=(Decimal("0.001") if model_id == "a" else Decimal("0.002")),
            mean_latency_ms=(1000.0 if model_id == "a" else 2000.0),
            cost_observations=trials,
            latency_observations=trials,
        ),
        masks=TargetMasks(
            cost=True,
            latency=True,
        ),
        metadata=MLMetadata(
            source="synthetic",
            task_id=f"task_{index}",
            domain="math",
            family_id=None,
            original_split=split,
            group_key=f"task:synthetic:task_{index}",
            run_ids=[1],
        ),
    )


def make_examples(start, stop, split):
    return [
        make_example(index, model_id, split)
        for index in range(start, stop)
        for model_id in ["a", "b"]
    ]


def write_examples(path, examples):
    path.write_text(
        "\n".join(item.model_dump_json() for item in examples) + "\n",
        encoding="utf-8",
    )


class FakePredictor:
    def __init__(self):
        self.preprocessing = {
            "models": ["a", "b"],
            "efforts": ["medium"],
        }

    def predict_many(self, requests):
        return [
            {
                "model_id": model_id,
                "effort": effort,
                "success_probability": (0.6 if model_id == "a" else 0.9),
                "predicted_cost_usd": (0.001 if model_id == "a" else 0.002),
                "predicted_latency_ms": (1000 if model_id == "a" else 2000),
            }
            for _, model_id, effort in requests
        ]


@pytest.fixture
def evaluation_data(tmp_path):
    train = make_examples(0, 10, "train")
    validation = make_examples(10, 14, "validation")
    test = make_examples(20, 24, "test")

    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    report_path = tmp_path / "report.json"

    write_examples(train_path, train)
    write_examples(validation_path, validation)
    write_examples(test_path, test)

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

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "train_path": train_path,
        "validation_path": validation_path,
        "test_path": test_path,
        "report_path": report_path,
    }


def test_probability_metrics(evaluation_data):
    examples = evaluation_data["test"]

    probabilities = [0.6 if item.features.model_id == "a" else 0.9 for item in examples]

    report = probability_report(
        examples,
        probabilities,
    )

    assert report["trials"] == 16
    assert report["failures"] == 4
    assert report["log_loss"] > 0
    assert report["brier_score"] >= 0
    assert report["expected_calibration_error"] >= 0


def test_pairwise_ranking(evaluation_data):
    examples = evaluation_data["test"]

    probabilities = [0.6 if item.features.model_id == "a" else 0.9 for item in examples]

    report = pairwise_ranking_report(
        examples,
        probabilities,
    )

    assert report["comparable_pairs"] == 4
    assert report["ranking_accuracy"] == 1.0


def test_calibration_rejects_one_class(evaluation_data):
    examples = [
        item for item in evaluation_data["test"] if item.features.model_id == "b"
    ]

    with pytest.raises(
        ValueError,
        match="Insufficient successes or failures",
    ):
        fit_temperature(
            examples,
            [0.8] * len(examples),
            min_groups=2,
            min_outcomes_per_class=1,
        )


def test_temperature_calibration(evaluation_data):
    examples = evaluation_data["test"]

    predictions = [0.4 if item.features.model_id == "a" else 0.8 for item in examples]

    calibrator, report = fit_temperature(
        examples,
        predictions,
        min_groups=2,
        min_outcomes_per_class=1,
    )

    assert calibrator.temperature > 0

    assert report["log_loss_after"] <= report["log_loss_before"] + 1e-8

    assert all(0 < value < 1 for value in calibrator.transform_many(predictions))


def test_policy_replay(evaluation_data):
    train = evaluation_data["train"]
    test = evaluation_data["test"]

    predictions = FakePredictor().predict_many(
        [
            (
                item.features.prompt,
                item.features.model_id,
                item.features.effort,
            )
            for item in test
        ]
    )

    report = replay_policies(
        train,
        test,
        predictions,
        CONFIGS,
        RoutingPolicy(
            objective="quality",
            min_success_probability=0.0,
        ),
    )

    assert report["task_coverage"]["complete_matrix_tasks"] == 4
    assert report["task_coverage"]["routed_tasks"] == 4

    assert report["policies"]["neural"]["mean_empirical_success"] == 1.0

    assert report["policies"]["random_uniform"][
        "mean_empirical_success"
    ] == pytest.approx(0.75)

    assert report["policies"]["fixed_cheapest"]["mean_empirical_success"] == 0.5

    assert report["policies"]["oracle"]["mean_empirical_success"] == 1.0


def test_incomplete_matrix_is_excluded(evaluation_data):
    test = [
        item
        for item in evaluation_data["test"]
        if not (item.metadata.task_id == "task_20" and item.features.model_id == "b")
    ]

    predictions = FakePredictor().predict_many(
        [
            (
                item.features.prompt,
                item.features.model_id,
                item.features.effort,
            )
            for item in test
        ]
    )

    report = replay_policies(
        evaluation_data["train"],
        test,
        predictions,
        CONFIGS,
        RoutingPolicy(objective="quality"),
    )

    assert report["task_coverage"]["excluded_incomplete_matrix"] == 1

    assert report["task_coverage"]["complete_matrix_tasks"] == 3


def test_router_abstentions(evaluation_data):
    test = evaluation_data["test"]

    predictions = FakePredictor().predict_many(
        [
            (
                item.features.prompt,
                item.features.model_id,
                item.features.effort,
            )
            for item in test
        ]
    )

    report = replay_policies(
        evaluation_data["train"],
        test,
        predictions,
        CONFIGS,
        RoutingPolicy(
            objective="quality",
            min_success_probability=0.99,
        ),
    )

    assert report["task_coverage"]["router_abstentions"] == 4

    assert report["task_coverage"]["routed_tasks"] == 0

    assert report["policies"]["neural"] is None


def test_development_evaluation(evaluation_data):
    data = evaluation_data

    report = evaluate_router(
        predictor=FakePredictor(),
        model_report_path=data["report_path"],
        train_path=data["train_path"],
        validation_path=data["validation_path"],
        evaluation_path=data["validation_path"],
        candidates=CONFIGS,
        policy=RoutingPolicy(
            objective="quality",
            min_success_probability=0.0,
        ),
        mode="development",
    )

    assert report["evaluation_mode"] == "development"
    assert report["dataset"]["tasks"] == 4

    assert report["automatic_execution_approved"] is False

    assert any("Development diagnostic" in warning for warning in report["warnings"])


def test_heldout_evaluation(evaluation_data):
    data = evaluation_data

    report = evaluate_router(
        predictor=FakePredictor(),
        model_report_path=data["report_path"],
        train_path=data["train_path"],
        validation_path=data["validation_path"],
        evaluation_path=data["test_path"],
        candidates=CONFIGS,
        policy=RoutingPolicy(
            objective="quality",
            min_success_probability=0.0,
        ),
        mode="heldout",
    )

    assert report["evaluation_mode"] == "heldout"
    assert report["dataset"]["tasks"] == 4

    assert report["policy_replay"]["task_coverage"]["routed_tasks"] == 4


def test_heldout_rejects_validation_overlap(
    evaluation_data,
):
    data = evaluation_data

    with pytest.raises(
        ValueError,
        match="Task-group leakage",
    ):
        evaluate_router(
            predictor=FakePredictor(),
            model_report_path=data["report_path"],
            train_path=data["train_path"],
            validation_path=data["validation_path"],
            evaluation_path=data["validation_path"],
            candidates=CONFIGS,
            policy=RoutingPolicy(),
            mode="heldout",
        )
