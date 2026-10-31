import hashlib
import json
from pathlib import Path

from backend.evaluation.calibration import fit_temperature
from backend.evaluation.metrics import (
    pairwise_ranking_report,
    probability_report,
    regression_report,
)
from backend.evaluation.replay import replay_policies
from backend.ml.data import (
    load_ml_examples,
    validate_training_splits,
)
from backend.routing.schemas import (
    CandidateConfig,
    RoutingPolicy,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_keys(examples) -> set[str]:
    return {example.metadata.group_key for example in examples}


def ensure_disjoint(
    left,
    right,
    left_name: str,
    right_name: str,
) -> None:
    overlap = group_keys(left) & group_keys(right)

    if overlap:
        raise ValueError(
            f"Task-group leakage between "
            f"{left_name} and {right_name}: "
            f"{sorted(overlap)[:5]}"
        )


def evaluate_router(
    *,
    predictor,
    model_report_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    evaluation_path: str | Path,
    candidates: list[CandidateConfig],
    policy: RoutingPolicy,
    mode: str = "development",
    calibration_path: str | Path | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict:

    if mode not in {"development", "heldout"}:
        raise ValueError("Invalid evaluation mode")

    train_path = Path(train_path)
    validation_path = Path(validation_path)
    evaluation_path = Path(evaluation_path)

    train = load_ml_examples(train_path)
    validation = load_ml_examples(validation_path)
    evaluation = load_ml_examples(evaluation_path)

    validate_training_splits(train, validation)

    model_report = json.loads(Path(model_report_path).read_text(encoding="utf-8"))

    if model_report["model_type"] != ("pytorch_multitask_router"):
        raise ValueError("Expected Phase 3.3 neural-router model")

    if model_report["dataset"]["train_sha256"] != file_hash(train_path) or model_report[
        "dataset"
    ]["validation_sha256"] != file_hash(validation_path):
        raise ValueError("Saved model training dataset hashes mismatch")

    if mode == "development":
        if file_hash(evaluation_path) != file_hash(validation_path):
            raise ValueError(
                "Development mode must evaluate the "
                "original early-stopping validation data"
            )

        if calibration_path is not None:
            raise ValueError(
                "Calibration is unavailable in development-diagnostic mode"
            )

    else:
        ensure_disjoint(
            train,
            evaluation,
            "train",
            "heldout",
        )

        ensure_disjoint(
            validation,
            evaluation,
            "validation",
            "heldout",
        )

        if any(item.metadata.original_split != "test" for item in evaluation):
            raise ValueError("Held-out mode requires original test split examples")

    candidate_keys = {(item.model_id, item.effort) for item in candidates}

    if not candidates or len(candidate_keys) != len(candidates):
        raise ValueError("Candidate configurations must be unique")

    supported_models = set(predictor.preprocessing["models"])

    supported_efforts = set(predictor.preprocessing["efforts"])

    trained_keys = {
        (
            example.features.model_id,
            example.features.effort,
        )
        for example in train
    }

    for key in candidate_keys:
        if (
            key[0] not in supported_models
            or key[1] not in supported_efforts
            or key not in trained_keys
        ):
            raise ValueError(f"Unsupported trained configuration: {key}")

    evaluation = [
        item
        for item in evaluation
        if (
            item.features.model_id,
            item.features.effort,
        )
        in candidate_keys
    ]

    if not evaluation:
        raise ValueError("No evaluation examples match the candidates")

    def predict_examples(examples):
        return predictor.predict_many(
            [
                (
                    item.features.prompt,
                    item.features.model_id,
                    item.features.effort,
                )
                for item in examples
            ]
        )

    predictions = predict_examples(evaluation)

    if len(predictions) != len(evaluation):
        raise ValueError("Predictor returned the wrong number of rows")

    for example, prediction in zip(
        evaluation,
        predictions,
        strict=True,
    ):
        expected = (
            example.features.model_id,
            example.features.effort,
        )

        actual = (
            prediction["model_id"],
            prediction["effort"],
        )

        if expected != actual:
            raise ValueError("Predictor output order/configuration mismatch")

    calibration_report = {
        "status": "not_applied",
    }

    # ----------------------------------
    # Optional independent calibration
    # ----------------------------------

    if calibration_path is not None:
        calibration_path = Path(calibration_path)

        calibration = load_ml_examples(calibration_path)

        ensure_disjoint(
            train,
            calibration,
            "train",
            "calibration",
        )

        ensure_disjoint(
            validation,
            calibration,
            "validation",
            "calibration",
        )

        ensure_disjoint(
            calibration,
            evaluation,
            "calibration",
            "evaluation",
        )

        if any(
            item.metadata.original_split
            in {
                "test",
                "smoke",
            }
            for item in calibration
        ):
            raise ValueError(
                "Calibration data must not consume official test or smoke tasks"
            )

        calibration = [
            item
            for item in calibration
            if (
                item.features.model_id,
                item.features.effort,
            )
            in candidate_keys
        ]

        calibration_predictions = predict_examples(calibration)

        calibrator, fit_report = fit_temperature(
            calibration,
            [item["success_probability"] for item in calibration_predictions],
        )

        predictions = [
            {
                **prediction,
                "success_probability": calibrator.transform(
                    prediction["success_probability"]
                ),
            }
            for prediction in predictions
        ]

        calibration_report = {
            "status": "applied",
            "calibration_sha256": file_hash(calibration_path),
            **fit_report,
            "artifact": calibrator.model_dump(),
        }

    # ----------------------------------
    # Predictive performance
    # ----------------------------------

    probabilities = [item["success_probability"] for item in predictions]

    prediction_metrics = {
        "success": probability_report(
            evaluation,
            probabilities,
        ),
        "pairwise_ranking": pairwise_ranking_report(
            evaluation,
            probabilities,
        ),
        "cost": regression_report(
            evaluation,
            predictions,
            target_field="mean_cost_usd",
            prediction_field="predicted_cost_usd",
            mask_field="cost",
        ),
        "latency": regression_report(
            evaluation,
            predictions,
            target_field="mean_latency_ms",
            prediction_field="predicted_latency_ms",
            mask_field="latency",
        ),
    }

    # ----------------------------------
    # Offline policy replay
    # ----------------------------------

    replay = replay_policies(
        train,
        evaluation,
        predictions,
        candidates,
        policy,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    observed_successes = sum(item.targets.successes for item in evaluation)

    total_trials = sum(item.targets.trials for item in evaluation)

    warnings = []

    if mode == "development":
        warnings.append(
            "Development diagnostic: the evaluation "
            "data was used for model early stopping. "
            "These are not independent test metrics."
        )

    if observed_successes in {
        0,
        total_trials,
    }:
        warnings.append(
            "Only one observed success/failure class; "
            "task discrimination and calibration "
            "cannot be established."
        )

    if len(group_keys(evaluation)) < 20:
        warnings.append(
            "Fewer than 20 evaluation task groups; estimates are highly uncertain."
        )

    if calibration_report["status"] != "applied":
        warnings.append("Success probabilities have not been independently calibrated.")

    return {
        "schema_version": 1,
        "evaluation_mode": mode,
        "dataset": {
            "evaluation_sha256": file_hash(evaluation_path),
            "examples": len(evaluation),
            "tasks": len(
                {
                    (
                        item.metadata.source,
                        item.metadata.task_id,
                    )
                    for item in evaluation
                }
            ),
            "groups": len(group_keys(evaluation)),
            "trials": total_trials,
            "successes": observed_successes,
            "failures": total_trials - observed_successes,
        },
        "policy": policy.model_dump(mode="json"),
        "calibration": calibration_report,
        "prediction_metrics": prediction_metrics,
        "policy_replay": replay,
        "warnings": warnings,
        "automatic_execution_approved": False,
    }
