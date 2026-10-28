import json
from decimal import Decimal
from pathlib import Path

import pytest

from backend.benchmarks.ml_dataset import (
    MLExample,
    MLMetadata,
    MLTargets,
    TargetMasks,
    extract_features,
)
from backend.ml.baseline import (
    load_baseline,
    make_matrix,
    predict_success,
    train_baseline,
)
from backend.ml.data import (
    load_ml_examples,
    validate_training_splits,
)
from backend.ml.features import FeatureEncoder
from backend.ml.metrics import (
    configuration_prior_predictions,
    evaluate_probabilities,
    global_prior_predictions,
)


def make_example(
    task_index: int,
    model_id: str,
    effort: str,
) -> MLExample:

    prompt = f"Solve problem {task_index}. What is {task_index} + {task_index + 3}?"

    trials = 4

    # Deterministic synthetic labels, with
    # both successes and failures.
    successes = task_index % 3 + int(effort == "high") + int(model_id == "model-b")

    successes = min(successes, trials)

    return MLExample(
        features=extract_features(
            prompt=prompt,
            model_id=model_id,
            effort=effort,
        ),
        targets=MLTargets(
            success_rate=successes / trials,
            successes=successes,
            trials=trials,
            mean_cost_usd=Decimal("0.001"),
            mean_latency_ms=100.0,
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
            original_split="train",
            group_key=(f"task:synthetic:task_{task_index}"),
            run_ids=[1],
        ),
    )


def make_examples(
    start: int,
    stop: int,
) -> list[MLExample]:

    return [
        make_example(index, model_id, effort)
        for index in range(start, stop)
        for model_id in ["model-a", "model-b"]
        for effort in ["medium", "high"]
    ]


def write_examples(
    path: Path,
    examples: list[MLExample],
) -> None:

    path.write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def training_data(tmp_path):

    train = make_examples(0, 24)
    validation = make_examples(24, 32)

    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"

    write_examples(train_path, train)
    write_examples(validation_path, validation)

    return {
        "train": train,
        "validation": validation,
        "train_path": train_path,
        "validation_path": validation_path,
        "output_dir": tmp_path / "model",
    }


def test_load_training_data(training_data):

    examples = load_ml_examples(training_data["train_path"])

    assert len(examples) == 96

    assert all(example.targets.trials == 4 for example in examples)


def test_feature_encoder(training_data):

    encoder = FeatureEncoder.fit(training_data["train"])

    assert encoder.models == [
        "model-a",
        "model-b",
    ]

    assert encoder.efforts == [
        "high",
        "medium",
    ]

    features = [example.features for example in training_data["train"][:3]]

    matrix = encoder.transform(features)

    assert matrix.shape == (
        3,
        len(encoder.feature_names),
    )


def test_unseen_model_rejected(training_data):

    encoder = FeatureEncoder.fit(training_data["train"])

    unknown = extract_features(
        prompt="Solve this problem.",
        model_id="unknown-model",
        effort="medium",
    )

    with pytest.raises(
        ValueError,
        match="Unseen model",
    ):
        encoder.transform([unknown])


def test_trial_weights(training_data):

    examples = training_data["train"]

    encoder = FeatureEncoder.fit(examples)

    matrix = make_matrix(
        examples,
        encoder,
    )

    weights = matrix.get_weight()

    assert len(weights) == 96
    assert all(weight == 4 for weight in weights)


def test_probability_metrics(training_data):

    examples = training_data["validation"]

    probabilities = [0.5] * len(examples)

    metrics = evaluate_probabilities(
        examples,
        probabilities,
    )

    assert metrics["examples"] == 32

    assert metrics["trials"] == 128

    assert metrics["log_loss"] == pytest.approx(
        0.693147,
        abs=1e-5,
    )

    assert metrics["brier_score"] == pytest.approx(0.25)


def test_prior_baselines(training_data):

    train = training_data["train"]
    validation = training_data["validation"]

    global_predictions = global_prior_predictions(
        train,
        validation,
    )

    configuration_predictions = configuration_prior_predictions(
        train,
        validation,
    )

    assert len(global_predictions) == 32
    assert len(configuration_predictions) == 32

    assert all(0 < p < 1 for p in global_predictions)

    assert all(0 < p < 1 for p in configuration_predictions)


def test_group_leakage_rejected(training_data):

    train = training_data["train"]
    validation = training_data["validation"]

    # Deliberately insert a training task into validation.
    contaminated = [
        *validation,
        train[0],
    ]

    with pytest.raises(
        ValueError,
        match="Task-group leakage",
    ):
        validate_training_splits(
            train,
            contaminated,
        )


def test_single_outcome_rejected(training_data):

    train = training_data["train"]
    validation = training_data["validation"]

    all_successful = []

    for example in train:
        all_successful.append(
            example.model_copy(
                update={
                    "targets": example.targets.model_copy(
                        update={
                            "success_rate": 1.0,
                            "successes": (example.targets.trials),
                        }
                    )
                }
            )
        )

    with pytest.raises(
        ValueError,
        match="only one outcome class",
    ):
        validate_training_splits(
            all_successful,
            validation,
        )


def test_training_creates_artifacts(training_data):

    report = train_baseline(
        train_path=training_data["train_path"],
        validation_path=(training_data["validation_path"]),
        output_dir=training_data["output_dir"],
        num_boost_round=30,
        early_stopping_rounds=5,
    )

    output = training_data["output_dir"]

    assert (output / "model.json").is_file()
    assert (output / "encoder.json").is_file()
    assert (output / "report.json").is_file()

    assert report["model_type"] == ("xgboost_success_baseline")

    assert report["dataset"]["train_examples"] == 96
    assert report["dataset"]["validation_examples"] == 32

    assert "xgboost" in report["metrics"]
    assert "global_prior" in report["metrics"]
    assert "configuration_prior" in report["metrics"]

    saved_report = json.loads((output / "report.json").read_text(encoding="utf-8"))

    assert saved_report["schema_version"] == 1


def test_saved_model_prediction(training_data):

    output = training_data["output_dir"]

    train_baseline(
        train_path=training_data["train_path"],
        validation_path=(training_data["validation_path"]),
        output_dir=output,
        num_boost_round=30,
        early_stopping_rounds=5,
    )

    booster, encoder, report = load_baseline(output)

    assert booster is not None
    assert encoder is not None
    assert report["training"]["best_iteration"] >= 0

    probability = predict_success(
        model_dir=output,
        prompt="What is 17 + 24?",
        model_id="model-a",
        effort="medium",
    )

    assert 0 <= probability <= 1


def test_missing_dataset_rejected(tmp_path):

    with pytest.raises(FileNotFoundError):
        load_ml_examples(tmp_path / "missing.jsonl")
