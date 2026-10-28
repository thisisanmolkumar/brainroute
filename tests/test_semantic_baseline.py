import hashlib
from decimal import Decimal

import numpy as np
import pytest

from backend.benchmarks.ml_dataset import (
    MLExample,
    MLMetadata,
    MLTargets,
    TargetMasks,
    extract_features,
)
from backend.ml.baseline import train_baseline
from backend.ml.semantic_baseline import (
    SemanticSuccessPredictor,
    train_semantic_baseline,
)
from backend.ml.semantic_features import (
    PromptPCA,
    SentenceEmbeddingExtractor,
)


class FakeSentenceModel:
    def __init__(self):
        self.encoded_batches = []

    def get_sentence_embedding_dimension(self):
        return 12

    def encode(self, prompts, **kwargs):
        self.encoded_batches.append(list(prompts))

        vectors = []

        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()

            vector = (
                np.asarray(
                    list(digest[:12]),
                    dtype=np.float32,
                )
                / 255.0
            )

            if kwargs.get("normalize_embeddings"):
                norm = np.linalg.norm(vector)
                vector = vector / max(norm, 1e-8)

            vectors.append(vector)

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


def make_examples(start, stop):
    examples = []

    for index in range(start, stop):
        for effort in ["medium", "high"]:
            trials = 4

            successes = index % 4 + int(effort == "high")

            prompt = f"Solve problem {index}. Find the result of {index} + {index + 3}."

            examples.append(
                MLExample(
                    features=extract_features(
                        prompt=prompt,
                        model_id="model-a",
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
                        task_id=f"task_{index}",
                        domain="math",
                        family_id=None,
                        original_split="train",
                        group_key=f"task:synthetic:task_{index}",
                        run_ids=[1],
                    ),
                )
            )

    return examples


def write_examples(path, examples):
    path.write_text(
        "\n".join(example.model_dump_json() for example in examples) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def training_setup(tmp_path):
    train = make_examples(0, 12)
    validation = make_examples(12, 16)

    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"

    write_examples(train_path, train)
    write_examples(validation_path, validation)

    baseline_dir = tmp_path / "baseline"
    semantic_dir = tmp_path / "semantic"

    # Train the original 3.1 baseline on exactly
    # the same synthetic data.
    train_baseline(
        train_path=train_path,
        validation_path=validation_path,
        output_dir=baseline_dir,
        num_boost_round=25,
        early_stopping_rounds=5,
        allow_small_data=True,
    )

    fake_model = FakeSentenceModel()

    embedder = SentenceEmbeddingExtractor(
        model_name="test-encoder",
        revision="test-revision",
        model=fake_model,
    )

    return {
        "train": train,
        "train_path": train_path,
        "validation_path": validation_path,
        "baseline_dir": baseline_dir,
        "semantic_dir": semantic_dir,
        "embedder": embedder,
        "fake_model": fake_model,
    }


def test_embedding_shape():
    embedder = SentenceEmbeddingExtractor(
        model=FakeSentenceModel(),
    )

    vectors = embedder.encode(
        [
            "What is 2 + 2?",
            "Explain binary search.",
        ]
    )

    assert vectors.shape == (2, 12)
    assert np.isfinite(vectors).all()


def test_duplicate_prompt_encoding():
    model = FakeSentenceModel()

    embedder = SentenceEmbeddingExtractor(
        model=model,
    )

    vectors = embedder.encode(
        [
            "Same question",
            "Different question",
            "Same question",
        ]
    )

    assert vectors.shape == (3, 12)

    # Only two distinct prompts sent to the model.
    assert len(model.encoded_batches[0]) == 2

    np.testing.assert_array_equal(
        vectors[0],
        vectors[2],
    )


def test_pca_roundtrip(tmp_path):
    embedder = SentenceEmbeddingExtractor(
        model=FakeSentenceModel(),
    )

    vectors = embedder.encode([f"Question number {index}" for index in range(10)])

    pca = PromptPCA.fit(
        vectors,
        dimensions=4,
    )

    reduced = pca.transform(vectors)

    assert reduced.shape == (10, 4)

    path = tmp_path / "pca.npz"
    pca.save(path)

    restored = PromptPCA.load(path)

    np.testing.assert_allclose(
        reduced,
        restored.transform(vectors),
        rtol=1e-5,
        atol=1e-6,
    )


def test_semantic_training(training_setup):
    setup = training_setup

    report = train_semantic_baseline(
        train_path=setup["train_path"],
        validation_path=setup["validation_path"],
        baseline_dir=setup["baseline_dir"],
        output_dir=setup["semantic_dir"],
        revision="test-revision",
        pca_dimensions=4,
        rounds=25,
        early_stopping_rounds=5,
        embedder=setup["embedder"],
    )

    output = setup["semantic_dir"]

    for filename in [
        "model.json",
        "encoder.json",
        "pca.npz",
        "embedding_config.json",
        "report.json",
    ]:
        assert (output / filename).is_file()

    assert report["model_type"] == ("xgboost_semantic_baseline")

    assert report["embedding"]["pca_dimensions"] == 4

    assert "semantic_xgboost" in report["metrics"]
    assert "phase_3_1_xgboost" in report["metrics"]


def test_semantic_inference(training_setup):
    setup = training_setup

    train_semantic_baseline(
        train_path=setup["train_path"],
        validation_path=setup["validation_path"],
        baseline_dir=setup["baseline_dir"],
        output_dir=setup["semantic_dir"],
        revision="test-revision",
        pca_dimensions=4,
        rounds=25,
        early_stopping_rounds=5,
        embedder=setup["embedder"],
    )

    predictor = SemanticSuccessPredictor(
        model_dir=setup["semantic_dir"],
        embedder=setup["embedder"],
    )

    predictions = predictor.predict_many(
        [
            ("Calculate 12 + 14.", "model-a", "medium"),
            ("Calculate 12 + 14.", "model-a", "high"),
        ]
    )

    assert len(predictions) == 2
    assert all(0 <= probability <= 1 for probability in predictions)

    with pytest.raises(ValueError, match="Unseen model"):
        predictor.predict(
            "Calculate 12 + 14.",
            "unknown-model",
            "medium",
        )


def test_mismatched_baseline_rejected(training_setup):
    setup = training_setup

    # Change the validation file after the
    # Phase 3.1 baseline was trained.
    path = setup["validation_path"]

    path.write_text(
        path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="different dataset files",
    ):
        train_semantic_baseline(
            train_path=setup["train_path"],
            validation_path=path,
            baseline_dir=setup["baseline_dir"],
            output_dir=setup["semantic_dir"],
            revision="test-revision",
            embedder=setup["embedder"],
        )
