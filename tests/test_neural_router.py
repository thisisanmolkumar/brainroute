import hashlib
import json
from decimal import Decimal

import numpy as np
import pytest
import torch

from backend.benchmarks.ml_dataset import (
    MLExample,
    MLMetadata,
    MLTargets,
    TargetMasks,
    extract_features,
)
from backend.ml.neural_router import (
    NeuralRouter,
    multitask_loss,
)
from backend.ml.neural_training import (
    NeuralRouterPredictor,
    train_neural_router,
)


class FakeEmbedder:
    model_name = "test-encoder"
    revision = "test-revision"
    dimension = 12

    def encode(self, prompts, batch_size=8):
        unique = {}

        for prompt in prompts:
            if prompt in unique:
                continue

            digest = hashlib.sha256(prompt.encode("utf-8")).digest()

            vector = (
                np.asarray(
                    list(digest[:12]),
                    dtype=np.float32,
                )
                / 255.0
            )

            vector /= max(
                np.linalg.norm(vector),
                1e-8,
            )

            unique[prompt] = vector

        return np.stack([unique[prompt] for prompt in prompts])


def make_example(
    index: int,
    model_id: str,
    effort: str,
    *,
    missing_cost: bool = False,
) -> MLExample:

    trials = 4

    successes = min(
        trials,
        index % 4 + int(effort == "high") + int(model_id == "model-b"),
    )

    cost = Decimal("0.001") * (2 if effort == "high" else 1)

    return MLExample(
        features=extract_features(
            prompt=(f"Solve problem {index}. Calculate {index} + {index + 3}."),
            model_id=model_id,
            effort=effort,
        ),
        targets=MLTargets(
            success_rate=successes / trials,
            successes=successes,
            trials=trials,
            mean_cost_usd=(None if missing_cost else cost),
            mean_latency_ms=(200.0 if effort == "high" else 100.0),
            cost_observations=(0 if missing_cost else trials),
            latency_observations=trials,
        ),
        masks=TargetMasks(
            cost=not missing_cost,
            latency=True,
        ),
        metadata=MLMetadata(
            source="synthetic",
            task_id=f"task_{index}",
            domain="math",
            family_id=None,
            original_split="train",
            group_key=f"task:synthetic:task_{index}",
            run_ids=[1],
        ),
    )


def make_examples(
    start: int,
    stop: int,
    *,
    missing_cost: bool = False,
):
    return [
        make_example(
            index,
            model_id,
            effort,
            missing_cost=missing_cost,
        )
        for index in range(start, stop)
        for model_id in ["model-a", "model-b"]
        for effort in ["medium", "high"]
    ]


def write_examples(path, examples):
    path.write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n",
        encoding="utf-8",
    )


def train_fixture(
    tmp_path,
    *,
    missing_cost=False,
):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"

    embedding_config_path = tmp_path / "embedding_config.json"

    write_examples(
        train_path,
        make_examples(
            0,
            24,
            missing_cost=missing_cost,
        ),
    )

    write_examples(
        validation_path,
        make_examples(
            24,
            32,
            missing_cost=missing_cost,
        ),
    )

    embedding_config_path.write_text(
        json.dumps(
            {
                "model_name": "test-encoder",
                "revision": "test-revision",
                "embedding_dimension": 12,
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "neural"

    report = train_neural_router(
        train_path=train_path,
        validation_path=validation_path,
        embedding_config_path=embedding_config_path,
        output_dir=output_dir,
        epochs=8,
        patience=3,
        batch_size=16,
        embedder=FakeEmbedder(),
    )

    return output_dir, report


def test_neural_forward_shapes():
    model = NeuralRouter(
        embedding_dim=12,
        num_models=2,
        num_efforts=2,
    )

    outputs = model(
        torch.randn(4, 12),
        torch.randn(4, 7),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
    )

    assert len(outputs) == 3
    assert all(output.shape == (4,) for output in outputs)


def test_missing_cost_is_ignored():
    outputs = (
        torch.tensor([0.2, -0.3]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.5, 1.5]),
    )

    kwargs = {
        "outputs": outputs,
        "success_rate": torch.tensor([1.0, 0.0]),
        "trials": torch.tensor([2.0, 2.0]),
        "latency_target": torch.tensor([1.0, 1.0]),
        "cost_mask": torch.tensor([1.0, 0.0]),
        "latency_mask": torch.tensor([1.0, 1.0]),
    }

    first = multitask_loss(
        cost_target=torch.tensor([0.0, 0.0]),
        **kwargs,
    )

    second = multitask_loss(
        cost_target=torch.tensor([0.0, 9999.0]),
        **kwargs,
    )

    assert torch.allclose(
        first["cost"],
        second["cost"],
    )

    assert torch.allclose(
        first["total"],
        second["total"],
    )


def test_all_missing_regression_labels():
    outputs = (
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
    )

    losses = multitask_loss(
        outputs=outputs,
        success_rate=torch.tensor([1.0, 0.0]),
        trials=torch.tensor([3.0, 2.0]),
        cost_target=torch.zeros(2),
        latency_target=torch.zeros(2),
        cost_mask=torch.zeros(2),
        latency_mask=torch.zeros(2),
    )

    assert losses["cost"].item() == 0.0
    assert losses["latency"].item() == 0.0
    assert torch.isfinite(losses["total"])


def test_training_creates_artifacts(tmp_path):
    output_dir, report = train_fixture(tmp_path)

    for filename in [
        "model.pt",
        "preprocessing.json",
        "report.json",
        "history.json",
    ]:
        assert (output_dir / filename).is_file()

    assert report["model_type"] == ("pytorch_multitask_router")

    assert report["training"]["best_epoch"] >= 1
    assert report["dataset"]["train_groups"] == 24

    assert report["metrics"]["success"]["trials"] == 128


def test_saved_model_inference(tmp_path):
    output_dir, _ = train_fixture(tmp_path)

    predictor = NeuralRouterPredictor(
        output_dir,
        embedder=FakeEmbedder(),
    )

    results = predictor.predict_many(
        [
            ("Calculate 12 + 13.", "model-a", "medium"),
            ("Calculate 12 + 13.", "model-a", "high"),
            ("Calculate 12 + 13.", "model-b", "high"),
        ]
    )

    assert len(results) == 3

    for result in results:
        assert 0 <= result["success_probability"] <= 1
        assert result["predicted_cost_usd"] >= 0
        assert result["predicted_latency_ms"] >= 0


def test_unseen_model_rejected(tmp_path):
    output_dir, _ = train_fixture(tmp_path)

    predictor = NeuralRouterPredictor(
        output_dir,
        embedder=FakeEmbedder(),
    )

    with pytest.raises(
        ValueError,
        match="Unknown model or effort",
    ):
        predictor.predict(
            "Solve this task.",
            "unknown-model",
            "medium",
        )


def test_head_without_labels_is_disabled(tmp_path):
    output_dir, report = train_fixture(
        tmp_path,
        missing_cost=True,
    )

    assert report["target_coverage"]["train_cost_examples"] == 0

    predictor = NeuralRouterPredictor(
        output_dir,
        embedder=FakeEmbedder(),
    )

    prediction = predictor.predict(
        "Calculate 12 + 13.",
        "model-a",
        "medium",
    )

    assert prediction["predicted_cost_usd"] is None
    assert prediction["predicted_latency_ms"] is not None
