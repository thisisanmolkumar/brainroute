import hashlib
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from backend.benchmarks.ml_dataset import MLFeatures, extract_features
from backend.ml.data import (
    group_count,
    load_ml_examples,
    validate_training_splits,
)
from backend.ml.features import FeatureEncoder
from backend.ml.metrics import (
    configuration_prior_predictions,
    evaluate_probabilities,
    global_prior_predictions,
)

MIN_TRAIN_GROUPS = 20


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_matrix(
    examples,
    encoder: FeatureEncoder,
) -> xgb.DMatrix:

    features = [example.features for example in examples]

    X = encoder.transform(features)

    targets = np.asarray(
        [example.targets.success_rate for example in examples],
        dtype=np.float32,
    )

    weights = np.asarray(
        [example.targets.trials for example in examples],
        dtype=np.float32,
    )

    return xgb.DMatrix(
        X,
        label=targets,
        weight=weights,
        feature_names=encoder.feature_names,
    )


def predict_with_booster(
    booster: xgb.Booster,
    encoder: FeatureEncoder,
    features: list[MLFeatures],
    best_iteration: int,
) -> np.ndarray:

    X = encoder.transform(features)

    matrix = xgb.DMatrix(
        X,
        feature_names=encoder.feature_names,
    )

    return booster.predict(
        matrix,
        iteration_range=(
            0,
            best_iteration + 1,
        ),
    )


def train_baseline(
    train_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
    allow_small_data: bool = False,
) -> dict:

    train_path = Path(train_path)
    validation_path = Path(validation_path)
    output_dir = Path(output_dir)

    if num_boost_round < 1:
        raise ValueError("num_boost_round must be positive")

    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be positive")

    # ----------------------------------
    # 1. Load and validate the dataset
    # ----------------------------------

    train = load_ml_examples(train_path)
    validation = load_ml_examples(validation_path)

    validate_training_splits(
        train,
        validation,
    )

    train_groups = group_count(train)
    validation_groups = group_count(validation)

    if train_groups < MIN_TRAIN_GROUPS and not allow_small_data:
        raise ValueError(
            f"Only {train_groups} training task groups found. "
            f"At least {MIN_TRAIN_GROUPS} are required by "
            "our initial data-quality guard. "
            "Use --allow-small-data only for pipeline tests."
        )

    # ----------------------------------
    # 2. Fit encoder on training data ONLY
    # ----------------------------------

    encoder = FeatureEncoder.fit(train)

    # ----------------------------------
    # 3. Build XGBoost matrices
    # ----------------------------------

    dtrain = make_matrix(train, encoder)
    dvalidation = make_matrix(validation, encoder)

    # ----------------------------------
    # 4. Define the model
    # ----------------------------------

    params = {
        "objective": "reg:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "seed": seed,
        "nthread": 4,
    }

    # ----------------------------------
    # 5. Train
    # ----------------------------------

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[
            (dvalidation, "validation"),
        ],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    best_iteration = int(booster.best_iteration)

    # ----------------------------------
    # 6. Generate validation predictions
    # ----------------------------------

    validation_predictions = predict_with_booster(
        booster,
        encoder,
        [item.features for item in validation],
        best_iteration,
    )

    # ----------------------------------
    # 7. Compare against simple baselines
    # ----------------------------------

    global_predictions = global_prior_predictions(
        train,
        validation,
    )

    configuration_predictions = configuration_prior_predictions(
        train,
        validation,
    )

    xgb_metrics = evaluate_probabilities(
        validation,
        validation_predictions,
    )

    global_metrics = evaluate_probabilities(
        validation,
        global_predictions,
    )

    configuration_metrics = evaluate_probabilities(
        validation,
        configuration_predictions,
    )

    # ----------------------------------
    # 8. Save model artifacts
    # ----------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    booster.save_model(output_dir / "model.json")

    (output_dir / "encoder.json").write_text(
        encoder.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": 1,
        "model_type": "xgboost_success_baseline",
        "objective": "reg:logistic",
        "xgboost_version": xgb.__version__,
        "seed": seed,
        "dataset": {
            "train_path": str(train_path.resolve()),
            "validation_path": (str(validation_path.resolve())),
            "train_sha256": file_sha256(train_path),
            "validation_sha256": (file_sha256(validation_path)),
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_groups": train_groups,
            "validation_groups": validation_groups,
        },
        "training": {
            "parameters": params,
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": (early_stopping_rounds),
            "best_iteration": best_iteration,
        },
        "metrics": {
            "xgboost": xgb_metrics,
            "global_prior": global_metrics,
            "configuration_prior": (configuration_metrics),
        },
        "warnings": [],
    }

    if train_groups < MIN_TRAIN_GROUPS:
        report["warnings"].append(
            "Trained on fewer than 20 task groups. "
            "This model is for pipeline validation only."
        )

    if xgb_metrics["log_loss"] >= configuration_metrics["log_loss"]:
        report["warnings"].append(
            "XGBoost did not outperform the configuration-prior baseline on validation."
        )

    (output_dir / "report.json").write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def load_baseline(
    model_dir: str | Path,
) -> tuple[xgb.Booster, FeatureEncoder, dict]:

    model_dir = Path(model_dir)

    report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))

    if report.get("schema_version") != 1:
        raise ValueError("Unsupported baseline artifact version")

    encoder = FeatureEncoder.model_validate_json(
        (model_dir / "encoder.json").read_text(encoding="utf-8")
    )

    booster = xgb.Booster()

    booster.load_model(model_dir / "model.json")

    return booster, encoder, report


def predict_success(
    model_dir: str | Path,
    prompt: str,
    model_id: str,
    effort: str,
) -> float:
    """
    Predict probability of success for one
    task/model/effort configuration.
    """

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    booster, encoder, report = load_baseline(model_dir)

    features = extract_features(
        prompt=prompt,
        model_id=model_id,
        effort=effort,
    )

    best_iteration = report["training"]["best_iteration"]

    probabilities = predict_with_booster(
        booster,
        encoder,
        [features],
        best_iteration,
    )

    return float(probabilities[0])
