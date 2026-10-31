import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.benchmarks.ml_dataset import (
    MLExample,
    MLFeatures,
    extract_features,
)
from backend.ml.data import (
    group_count,
    load_ml_examples,
    validate_training_splits,
)
from backend.ml.features import NUMERIC_FIELDS
from backend.ml.metrics import evaluate_probabilities
from backend.ml.neural_router import (
    NeuralRouter,
    multitask_loss,
)
from backend.ml.semantic_features import (
    SentenceEmbeddingExtractor,
)

COST_SCALE = 1_000_000.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_matrix(
    features: list[MLFeatures],
) -> np.ndarray:
    values = np.asarray(
        [
            [float(getattr(item, field)) for field in NUMERIC_FIELDS]
            for item in features
        ],
        dtype=np.float32,
    )

    return np.log1p(values)


def weighted_stats(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 1.0

    mean = float(np.average(values, weights=weights))

    variance = float(
        np.average(
            (values - mean) ** 2,
            weights=weights,
        )
    )

    std = math.sqrt(variance)

    return mean, std if std > 1e-6 else 1.0


def fit_preprocessing(
    train: list[MLExample],
    embedding_dimension: int,
    embedding_model: str,
    embedding_revision: str,
) -> dict:
    features = [item.features for item in train]

    numbers = numeric_matrix(features)

    numeric_mean = numbers.mean(axis=0)
    numeric_std = numbers.std(axis=0)

    numeric_std = np.where(
        numeric_std > 1e-6,
        numeric_std,
        1.0,
    )

    cost_values = []
    cost_weights = []

    latency_values = []
    latency_weights = []

    for example in train:
        targets = example.targets

        if example.masks.cost:
            cost_values.append(math.log1p(
                float(targets.mean_cost_usd) * COST_SCALE))
            cost_weights.append(targets.trials)

        if example.masks.latency:
            latency_values.append(math.log1p(targets.mean_latency_ms))
            latency_weights.append(targets.trials)

    cost_mean, cost_std = weighted_stats(
        np.asarray(cost_values, dtype=np.float64),
        np.asarray(cost_weights, dtype=np.float64),
    )

    latency_mean, latency_std = weighted_stats(
        np.asarray(latency_values, dtype=np.float64),
        np.asarray(latency_weights, dtype=np.float64),
    )

    return {
        "schema_version": 1,
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "embedding_dimension": embedding_dimension,
        "models": sorted({item.features.model_id for item in train}),
        "efforts": sorted({item.features.effort for item in train}),
        "numeric_mean": numeric_mean.tolist(),
        "numeric_std": numeric_std.tolist(),
        "cost_mean": cost_mean,
        "cost_std": cost_std,
        "latency_mean": latency_mean,
        "latency_std": latency_std,
        "cost_trained": bool(cost_values),
        "latency_trained": bool(latency_values),
    }


def encode_inputs(
    features: list[MLFeatures],
    embedder: SentenceEmbeddingExtractor,
    preprocessing: dict,
) -> tuple[torch.Tensor, ...]:

    prompts = [item.prompt for item in features]

    embeddings = embedder.encode(
        prompts,
        batch_size=8,
    )

    expected_shape = (
        len(features),
        preprocessing["embedding_dimension"],
    )

    if embeddings.shape != expected_shape:
        raise ValueError("Unexpected sentence embedding dimensions")

    numbers = numeric_matrix(features)

    means = np.asarray(
        preprocessing["numeric_mean"],
        dtype=np.float32,
    )

    stds = np.asarray(
        preprocessing["numeric_std"],
        dtype=np.float32,
    )

    normalized = (numbers - means) / stds

    model_lookup = {value: index for index,
                    value in enumerate(preprocessing["models"])}

    effort_lookup = {
        value: index for index, value in enumerate(preprocessing["efforts"])
    }

    try:
        model_ids = [model_lookup[item.model_id] for item in features]

        effort_ids = [effort_lookup[item.effort] for item in features]

    except KeyError as exc:
        raise ValueError(f"Unknown model or effort: {exc}") from exc

    return (
        torch.tensor(
            embeddings,
            dtype=torch.float32,
        ),
        torch.tensor(
            normalized,
            dtype=torch.float32,
        ),
        torch.tensor(
            model_ids,
            dtype=torch.long,
        ),
        torch.tensor(
            effort_ids,
            dtype=torch.long,
        ),
    )


def make_dataset(
    examples: list[MLExample],
    embedder: SentenceEmbeddingExtractor,
    preprocessing: dict,
) -> TensorDataset:

    features = [example.features for example in examples]

    inputs = encode_inputs(
        features,
        embedder,
        preprocessing,
    )

    successes = []
    trials = []

    costs = []
    latencies = []

    cost_masks = []
    latency_masks = []

    for example in examples:
        target = example.targets

        successes.append(target.success_rate)
        trials.append(float(target.trials))

        valid_cost = example.masks.cost and preprocessing["cost_trained"]

        valid_latency = example.masks.latency and preprocessing["latency_trained"]

        cost_masks.append(float(valid_cost))
        latency_masks.append(float(valid_latency))

        costs.append(
            (
                math.log1p(float(target.mean_cost_usd) * COST_SCALE)
                - preprocessing["cost_mean"]
            )
            / preprocessing["cost_std"]
            if valid_cost
            else 0.0
        )

        latencies.append(
            (math.log1p(target.mean_latency_ms) -
             preprocessing["latency_mean"])
            / preprocessing["latency_std"]
            if valid_latency
            else 0.0
        )

    return TensorDataset(
        *inputs,
        torch.tensor(successes, dtype=torch.float32),
        torch.tensor(trials, dtype=torch.float32),
        torch.tensor(costs, dtype=torch.float32),
        torch.tensor(latencies, dtype=torch.float32),
        torch.tensor(cost_masks, dtype=torch.float32),
        torch.tensor(latency_masks, dtype=torch.float32),
    )


def batch_loss(
    model: NeuralRouter,
    batch: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:

    outputs = model(*batch[:4])

    return multitask_loss(
        outputs=outputs,
        success_rate=batch[4],
        trials=batch[5],
        cost_target=batch[6],
        latency_target=batch[7],
        cost_mask=batch[8],
        latency_mask=batch[9],
    )


def train_neural_router(
    train_path: str | Path,
    validation_path: str | Path,
    embedding_config_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    epochs: int = 200,
    patience: int = 25,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    embedder: SentenceEmbeddingExtractor | None = None,
) -> dict:

    train_path = Path(train_path)
    validation_path = Path(validation_path)
    embedding_config_path = Path(embedding_config_path)
    output_dir = Path(output_dir)

    if epochs < 1 or patience < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid training parameters")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------------------
    # 1. Load and verify training data
    # ----------------------------------

    train = load_ml_examples(train_path)
    validation = load_ml_examples(validation_path)

    validate_training_splits(train, validation)

    with embedding_config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        embedding_config = json.load(file)

    model_name = embedding_config["model_name"]
    revision = embedding_config["revision"]

    if not revision:
        raise ValueError("Embedding model revision must be pinned")

    print(
        "[3.3] Loading frozen SentenceTransformer...",
        flush=True,
    )

    if embedder is None:
        embedder = SentenceEmbeddingExtractor(
            model_name=model_name,
            revision=revision,
            device="cpu",
        )

    if embedder.dimension != embedding_config["embedding_dimension"]:
        raise ValueError("Embedding dimension mismatch")

    # ----------------------------------
    # 2. Fit preprocessing on train only
    # ----------------------------------

    preprocessing = fit_preprocessing(
        train,
        embedding_dimension=embedder.dimension,
        embedding_model=model_name,
        embedding_revision=revision,
    )

    # ----------------------------------
    # 3. Encode prompts ONCE
    # ----------------------------------

    print(
        "[3.3] Preparing training embeddings...",
        flush=True,
    )

    train_dataset = make_dataset(
        train,
        embedder,
        preprocessing,
    )

    print(
        "[3.3] Preparing validation embeddings...",
        flush=True,
    )

    validation_dataset = make_dataset(
        validation,
        embedder,
        preprocessing,
    )

    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    validation_batch = tuple(tensor for tensor in validation_dataset.tensors)

    # ----------------------------------
    # 4. Initialize network
    # ----------------------------------

    model = NeuralRouter(
        embedding_dim=embedder.dimension,
        num_models=len(preprocessing["models"]),
        num_efforts=len(preprocessing["efforts"]),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # ----------------------------------
    # 5. Training with early stopping
    # ----------------------------------

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            losses = batch_loss(model, batch)

            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        model.eval()

        with torch.inference_mode():
            validation_losses = batch_loss(
                model,
                validation_batch,
            )

        validation_loss = float(validation_losses["total"].item())

        if not math.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")

        history.append(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch

            best_state = copy.deepcopy(model.state_dict())

            bad_epochs = 0

        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[3.3] Epoch {epoch:03d} | "
                f"val_loss={validation_loss:.6f} | "
                f"best={best_loss:.6f}",
                flush=True,
            )

        if bad_epochs >= patience:
            print(
                f"[3.3] Early stopping at epoch {epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced")

    model.load_state_dict(best_state)
    model.eval()

    # ----------------------------------
    # 6. Evaluate best checkpoint
    # ----------------------------------

    with torch.inference_mode():
        outputs = model(*validation_batch[:4])

    success_logits, cost_z, latency_z = outputs

    success_predictions = torch.sigmoid(success_logits).cpu().numpy()

    success_metrics = evaluate_probabilities(
        validation,
        success_predictions,
    )

    def inverse_predictions(
        values: torch.Tensor,
        mean: float,
        std: float,
        scale: float = 1.0,
    ) -> np.ndarray:

        transformed = values.cpu().numpy() * std + mean

        # Prevent numerical overflow from extreme
        # untrained or out-of-distribution predictions.
        transformed = np.clip(
            transformed,
            0.0,
            20.0,
        )

        return np.expm1(transformed) / scale

    cost_predictions = inverse_predictions(
        cost_z,
        preprocessing["cost_mean"],
        preprocessing["cost_std"],
        COST_SCALE,
    )

    latency_predictions = inverse_predictions(
        latency_z,
        preprocessing["latency_mean"],
        preprocessing["latency_std"],
    )

    def regression_metrics(
        predictions: np.ndarray,
        target_name: str,
        mask_name: str,
        enabled: bool,
    ) -> dict:

        errors = []
        weights = []

        if enabled:
            for example, prediction in zip(
                validation,
                predictions,
                strict=True,
            ):
                if not getattr(
                    example.masks,
                    mask_name,
                ):
                    continue

                target = getattr(
                    example.targets,
                    target_name,
                )

                errors.append(abs(float(prediction) - float(target)))

                weights.append(example.targets.trials)

        if not errors:
            return {
                "observations": 0,
                "weighted_mae": None,
            }

        return {
            "observations": sum(weights),
            "weighted_mae": float(
                np.average(
                    errors,
                    weights=weights,
                )
            ),
        }

    cost_metrics = regression_metrics(
        cost_predictions,
        "mean_cost_usd",
        "cost",
        preprocessing["cost_trained"],
    )

    latency_metrics = regression_metrics(
        latency_predictions,
        "mean_latency_ms",
        "latency",
        preprocessing["latency_trained"],
    )

    # ----------------------------------
    # 7. Save artifacts and report
    # ----------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        output_dir / "model.pt",
    )

    (output_dir / "preprocessing.json").write_text(
        json.dumps(
            preprocessing,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": 1,
        "model_type": "pytorch_multitask_router",
        "seed": seed,
        "torch_version": torch.__version__,
        "dataset": {
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(validation_path),
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_groups": group_count(train),
            "validation_groups": group_count(validation),
        },
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "cost_loss_weight": 0.15,
            "latency_loss_weight": 0.15,
        },
        "target_coverage": {
            "train_cost_examples": sum(item.masks.cost for item in train),
            "train_latency_examples": sum(item.masks.latency for item in train),
        },
        "metrics": {
            "success": success_metrics,
            "cost": cost_metrics,
            "latency": latency_metrics,
        },
        "warnings": [],
    }

    if success_metrics["observed_success_rate"] in {
        0.0,
        1.0,
    }:
        report["warnings"].append(
            "Validation has only one success outcome "
            "class; routing discrimination is unproven."
        )

    if group_count(train) < 100:
        report["warnings"].append(
            "Fewer than 100 distinct training task "
            "groups; treat results as exploratory."
        )

    if not preprocessing["cost_trained"]:
        report["warnings"].append("Cost head has no valid training labels.")

    if not preprocessing["latency_trained"]:
        report["warnings"].append("Latency head has no valid training labels.")

    (output_dir / "report.json").write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (output_dir / "history.json").write_text(
        json.dumps(
            history,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


class NeuralRouterPredictor:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        embedder: SentenceEmbeddingExtractor | None = None,
    ) -> None:

        model_dir = Path(model_dir)

        self.preprocessing = json.loads(
            (model_dir / "preprocessing.json").read_text(encoding="utf-8")
        )

        if self.preprocessing["schema_version"] != 1:
            raise ValueError("Unsupported preprocessing version")

        if embedder is None:
            embedder = SentenceEmbeddingExtractor(
                model_name=self.preprocessing["embedding_model"],
                revision=self.preprocessing["embedding_revision"],
                device="cpu",
            )

        self.embedder = embedder

        if embedder.dimension != self.preprocessing["embedding_dimension"]:
            raise ValueError("Embedding dimension mismatch")

        self.model = NeuralRouter(
            embedding_dim=embedder.dimension,
            num_models=len(self.preprocessing["models"]),
            num_efforts=len(self.preprocessing["efforts"]),
        )

        state = torch.load(
            model_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(state)
        self.model.eval()

    def predict_many(
        self,
        requests: list[tuple[str, str, str]],
    ) -> list[dict]:

        if not requests:
            return []

        if any(not prompt.strip() for prompt, _, _ in requests):
            raise ValueError("Prompt cannot be empty")

        features = [
            extract_features(
                prompt=prompt,
                model_id=model_id,
                effort=effort,
            )
            for prompt, model_id, effort in requests
        ]

        inputs = encode_inputs(
            features,
            self.embedder,
            self.preprocessing,
        )

        with torch.inference_mode():
            success_logits, cost_z, latency_z = self.model(*inputs)

        success = torch.sigmoid(success_logits).numpy()

        cost_log = (
            cost_z.numpy() * self.preprocessing["cost_std"]
            + self.preprocessing["cost_mean"]
        )

        latency_log = (
            latency_z.numpy() * self.preprocessing["latency_std"]
            + self.preprocessing["latency_mean"]
        )

        costs = np.expm1(np.clip(cost_log, 0.0, 20.0)) / COST_SCALE

        latencies = np.expm1(np.clip(latency_log, 0.0, 20.0))

        results = []

        for index, (_, model_id, effort) in enumerate(requests):
            results.append(
                {
                    "model_id": model_id,
                    "effort": effort,
                    "success_probability": float(success[index]),
                    "predicted_cost_usd": (
                        float(costs[index])
                        if self.preprocessing["cost_trained"]
                        else None
                    ),
                    "predicted_latency_ms": (
                        float(latencies[index])
                        if self.preprocessing["latency_trained"]
                        else None
                    ),
                }
            )

        return results

    def predict(
        self,
        prompt: str,
        model_id: str,
        effort: str,
    ) -> dict:

        return self.predict_many([(prompt, model_id, effort)])[0]
