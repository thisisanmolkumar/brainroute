import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from backend.benchmarks.ml_dataset import (
    MLExample,
    extract_features,
)
from backend.ml.baseline import (
    file_sha256,
    load_baseline,
    predict_with_booster,
)
from backend.ml.data import (
    group_count,
    load_ml_examples,
    validate_training_splits,
)
from backend.ml.features import FeatureEncoder
from backend.ml.metrics import (
    configuration_prior_predictions,
    evaluate_probabilities,
)
from backend.ml.semantic_features import (
    MODEL_NAME,
    PromptPCA,
    SentenceEmbeddingExtractor,
)


def make_semantic_features(
    examples: list[MLExample],
    encoder: FeatureEncoder,
    embedder: SentenceEmbeddingExtractor,
    pca: PromptPCA,
) -> np.ndarray:

    features = [example.features for example in examples]

    tabular = encoder.transform(features)

    prompts = [item.prompt for item in features]

    embeddings = embedder.encode(prompts)

    reduced = pca.transform(embeddings)

    return np.concatenate(
        [tabular, reduced],
        axis=1,
    ).astype(np.float32)


def semantic_feature_names(
    encoder: FeatureEncoder,
    pca: PromptPCA,
) -> list[str]:

    return [
        *encoder.feature_names,
        *[f"semantic_{index:02d}" for index in range(len(pca.components))],
    ]


def make_dmatrix(
    examples: list[MLExample],
    matrix: np.ndarray,
    names: list[str],
) -> xgb.DMatrix:

    labels = np.asarray(
        [item.targets.success_rate for item in examples],
        dtype=np.float32,
    )

    weights = np.asarray(
        [item.targets.trials for item in examples],
        dtype=np.float32,
    )

    return xgb.DMatrix(
        matrix,
        label=labels,
        weight=weights,
        feature_names=names,
    )


def train_semantic_baseline(
    train_path: str | Path,
    validation_path: str | Path,
    baseline_dir: str | Path,
    output_dir: str | Path,
    *,
    revision: str,
    seed: int = 42,
    pca_dimensions: int = 16,
    rounds: int = 300,
    early_stopping_rounds: int = 30,
    embedder: SentenceEmbeddingExtractor | None = None,
) -> dict:

    train_path = Path(train_path)
    validation_path = Path(validation_path)
    output_dir = Path(output_dir)

    if not revision.strip():
        raise ValueError("An embedding revision is required")

    if rounds < 1 or early_stopping_rounds < 1:
        raise ValueError("Invalid training parameters")

    # ----------------------------------
    # 1. Load data
    # ----------------------------------

    train = load_ml_examples(train_path)
    validation = load_ml_examples(validation_path)

    validate_training_splits(train, validation)

    # ----------------------------------
    # 2. Load Phase 3.1 baseline
    # ----------------------------------

    old_booster, old_encoder, old_report = load_baseline(baseline_dir)

    if old_report["dataset"]["train_sha256"] != file_sha256(train_path) or old_report[
        "dataset"
    ]["validation_sha256"] != file_sha256(validation_path):
        raise ValueError(
            "Phase 3.1 baseline was trained/evaluated using different dataset files"
        )

    # ----------------------------------
    # 3. Initialize embedding model
    # ----------------------------------

    if embedder is None:
        embedder = SentenceEmbeddingExtractor(
            model_name=MODEL_NAME,
            revision=revision,
            device="cpu",
        )

    # ----------------------------------
    # 4. Fit tabular encoder on train only
    # ----------------------------------

    encoder = FeatureEncoder.fit(train)

    # ----------------------------------
    # 5. Fit PCA on unique training prompts
    # ----------------------------------

    train_prompts = list(dict.fromkeys(item.features.prompt for item in train))

    training_embeddings = embedder.encode(train_prompts)

    pca = PromptPCA.fit(
        training_embeddings,
        dimensions=pca_dimensions,
    )

    # ----------------------------------
    # 6. Build train/validation matrices
    # ----------------------------------

    X_train = make_semantic_features(
        train,
        encoder,
        embedder,
        pca,
    )

    X_validation = make_semantic_features(
        validation,
        encoder,
        embedder,
        pca,
    )

    feature_names = semantic_feature_names(
        encoder,
        pca,
    )

    dtrain = make_dmatrix(
        train,
        X_train,
        feature_names,
    )

    dvalidation = make_dmatrix(
        validation,
        X_validation,
        feature_names,
    )

    # ----------------------------------
    # 7. Train XGBoost
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

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[
            (dvalidation, "validation"),
        ],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    best_iteration = int(booster.best_iteration)

    # ----------------------------------
    # 8. Evaluate semantic model
    # ----------------------------------

    semantic_predictions = booster.predict(
        dvalidation,
        iteration_range=(0, best_iteration + 1),
    )

    semantic_metrics = evaluate_probabilities(
        validation,
        semantic_predictions,
    )

    # ----------------------------------
    # 9. Evaluate original baseline
    #    on the EXACT same validation data
    # ----------------------------------

    old_predictions = predict_with_booster(
        old_booster,
        old_encoder,
        [item.features for item in validation],
        old_report["training"]["best_iteration"],
    )

    old_metrics = evaluate_probabilities(
        validation,
        old_predictions,
    )

    prior_predictions = configuration_prior_predictions(
        train,
        validation,
    )

    prior_metrics = evaluate_probabilities(
        validation,
        prior_predictions,
    )

    # ----------------------------------
    # 10. Save artifacts
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

    pca.save(output_dir / "pca.npz")

    embedding_config = {
        "model_name": embedder.model_name,
        "revision": revision,
        "embedding_dimension": embedder.dimension,
        "pca_dimensions": len(pca.components),
        "normalized": True,
    }

    (output_dir / "embedding_config.json").write_text(
        json.dumps(embedding_config, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": 1,
        "model_type": "xgboost_semantic_baseline",
        "xgboost_version": xgb.__version__,
        "seed": seed,
        "dataset": {
            "train_sha256": file_sha256(train_path),
            "validation_sha256": file_sha256(validation_path),
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_groups": group_count(train),
            "validation_groups": group_count(validation),
        },
        "training": {
            "parameters": params,
            "rounds": rounds,
            "early_stopping_rounds": early_stopping_rounds,
            "best_iteration": best_iteration,
        },
        "embedding": embedding_config,
        "metrics": {
            "semantic_xgboost": semantic_metrics,
            "phase_3_1_xgboost": old_metrics,
            "configuration_prior": prior_metrics,
        },
        "warnings": [],
    }

    if semantic_metrics["observed_success_rate"] in {
        0.0,
        1.0,
    }:
        report["warnings"].append(
            "Validation contains only one observed "
            "outcome class. This cannot establish "
            "task-discrimination quality."
        )

    if semantic_metrics["log_loss"] >= old_metrics["log_loss"]:
        report["warnings"].append(
            "Semantic model did not improve validation log loss over Phase 3.1."
        )

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    return report


class SemanticSuccessPredictor:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        embedder: SentenceEmbeddingExtractor | None = None,
    ):
        model_dir = Path(model_dir)

        self.report = json.loads(
            (model_dir / "report.json").read_text(encoding="utf-8")
        )

        config = json.loads(
            (model_dir / "embedding_config.json").read_text(encoding="utf-8")
        )

        self.encoder = FeatureEncoder.model_validate_json(
            (model_dir / "encoder.json").read_text(encoding="utf-8")
        )

        self.pca = PromptPCA.load(model_dir / "pca.npz")

        self.embedder = (
            embedder
            if embedder is not None
            else SentenceEmbeddingExtractor(
                model_name=config["model_name"],
                revision=config["revision"],
                device=device,
            )
        )

        if self.embedder.dimension != config["embedding_dimension"]:
            raise ValueError("Embedding model dimension mismatch")

        if len(self.pca.components) != config["pca_dimensions"]:
            raise ValueError("Saved PCA dimension mismatch")

        self.booster = xgb.Booster()

        self.booster.load_model(model_dir / "model.json")

        self.feature_names = semantic_feature_names(
            self.encoder,
            self.pca,
        )

        self.best_iteration = self.report["training"]["best_iteration"]

    def predict_many(
        self,
        requests: list[tuple[str, str, str]],
    ) -> list[float]:

        if not requests:
            return []

        features = [
            extract_features(
                prompt=prompt,
                model_id=model_id,
                effort=effort,
            )
            for prompt, model_id, effort in requests
        ]

        tabular = self.encoder.transform(features)

        embeddings = self.embedder.encode([item.prompt for item in features])

        reduced = self.pca.transform(embeddings)

        matrix = np.concatenate(
            [tabular, reduced],
            axis=1,
        ).astype(np.float32)

        dmatrix = xgb.DMatrix(
            matrix,
            feature_names=self.feature_names,
        )

        probabilities = self.booster.predict(
            dmatrix,
            iteration_range=(
                0,
                self.best_iteration + 1,
            ),
        )

        return [float(value) for value in probabilities]

    def predict(
        self,
        prompt: str,
        model_id: str,
        effort: str,
    ) -> float:

        return self.predict_many([(prompt, model_id, effort)])[0]
