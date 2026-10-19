import json
from decimal import Decimal

import pytest

from backend.benchmarks.ml_dataset import (
    aggregate_executions,
    build_ml_dataset,
    load_collection_records,
    split_ml_dataset,
)


def make_records():
    """
    10 training tasks + 2 official test tasks.

    Each task has:
      - Medium reasoning
      - High reasoning
      - 2 repetitions

    Total: 12 × 2 × 2 = 48 records.
    """

    records = []

    for task_index in range(12):
        original_split = "train" if task_index < 10 else "test"

        for effort in ["medium", "high"]:
            for repeat_index in range(2):
                correct = task_index % 2 == 0 or effort == "high"

                records.append(
                    {
                        "run_id": 1,
                        "task_id": f"task_{task_index:02d}",
                        "source": "synthetic",
                        "domain": "math",
                        "split": original_split,
                        "family_id": None,
                        "prompt": (f"Solve synthetic task number {task_index}."),
                        "reference_answer": "42",
                        "model_id": "model-a",
                        "effort": effort,
                        "repeat_index": repeat_index,
                        "attempt_status": "completed",
                        "verification_status": ("correct" if correct else "incorrect"),
                        "strict_success": correct,
                        "cost_usd": ("0.001" if effort == "medium" else "0.002"),
                        "latency_ms": (100 if effort == "medium" else 200),
                    }
                )

    return records


def write_records(
    tmp_path,
    records,
    filename="collection.jsonl",
):
    path = tmp_path / filename

    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    return path


def prepare_test_dataset(tmp_path, records=None):
    if records is None:
        records = make_records()

    path = write_records(
        tmp_path,
        records,
    )

    return build_ml_dataset(
        input_paths=[path],
        output_dir=tmp_path / "ml",
        validation_fraction=0.2,
        seed=42,
    )


def read_ml_split(output_dir, split):
    path = output_dir / f"{split}.jsonl"

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_load_collection_records(tmp_path):
    path = write_records(
        tmp_path,
        make_records(),
    )

    records = load_collection_records([path])

    assert len(records) == 48


def test_repetition_aggregation(tmp_path):
    path = write_records(
        tmp_path,
        make_records(),
    )

    records = load_collection_records([path])

    examples, stats = aggregate_executions(records)

    # 12 tasks × 2 configurations
    assert len(examples) == 24

    assert stats["terminal_records_used"] == 48

    medium = next(
        example
        for example in examples
        if (
            example.metadata.task_id == "task_01"
            and example.features.effort == "medium"
        )
    )

    assert medium.targets.trials == 2
    assert medium.targets.successes == 0
    assert medium.targets.success_rate == 0.0

    high = next(
        example
        for example in examples
        if (example.metadata.task_id == "task_01" and example.features.effort == "high")
    )

    assert high.targets.trials == 2
    assert high.targets.successes == 2
    assert high.targets.success_rate == 1.0


def test_target_generation(tmp_path):
    path = write_records(
        tmp_path,
        make_records(),
    )

    records = load_collection_records([path])

    examples, _ = aggregate_executions(records)

    medium = next(
        example
        for example in examples
        if (
            example.metadata.task_id == "task_00"
            and example.features.effort == "medium"
        )
    )

    assert medium.targets.mean_latency_ms == 100

    assert medium.masks.cost is True
    assert medium.masks.latency is True
    assert medium.targets.mean_cost_usd == Decimal("0.001")


def test_group_aware_splitting(tmp_path):
    path = write_records(
        tmp_path,
        make_records(),
    )

    records = load_collection_records([path])

    examples, _ = aggregate_executions(records)

    splits = split_ml_dataset(
        examples,
        validation_fraction=0.2,
        seed=42,
    )

    # Ten train-origin tasks:
    # Eight training, two validation.
    #
    # Two official test tasks remain test.

    assert len(splits["train"]) == 16
    assert len(splits["validation"]) == 4
    assert len(splits["test"]) == 4

    split_groups = {
        split: {example.metadata.group_key for example in rows}
        for split, rows in splits.items()
    }

    assert not (split_groups["train"] & split_groups["validation"])

    assert not (split_groups["train"] & split_groups["test"])

    assert not (split_groups["validation"] & split_groups["test"])


def test_official_test_preserved(tmp_path):
    prepare_test_dataset(tmp_path)

    test_rows = read_ml_split(
        tmp_path / "ml",
        "test",
    )

    assert len(test_rows) == 4

    assert {row["metadata"]["task_id"] for row in test_rows} == {
        "task_10",
        "task_11",
    }


def test_different_models_stay_together(tmp_path):
    prepare_test_dataset(tmp_path)

    task_splits = {}

    for split in ["train", "validation", "test"]:
        rows = read_ml_split(
            tmp_path / "ml",
            split,
        )

        for row in rows:
            task_id = row["metadata"]["task_id"]

            task_splits.setdefault(
                task_id,
                set(),
            ).add(split)

    assert all(len(splits) == 1 for splits in task_splits.values())


def test_missing_cost_mask(tmp_path):
    records = make_records()

    # Remove cost/latency from one repetition.
    records[0]["cost_usd"] = None
    records[0]["latency_ms"] = None

    path = write_records(
        tmp_path,
        records,
    )

    loaded = load_collection_records([path])

    examples, _ = aggregate_executions(loaded)

    affected = next(
        example
        for example in examples
        if (
            example.metadata.task_id == "task_00"
            and example.features.effort == "medium"
        )
    )

    assert affected.targets.trials == 2

    assert affected.targets.mean_cost_usd is None
    assert affected.targets.mean_latency_ms is None

    assert affected.targets.cost_observations == 1
    assert affected.targets.latency_observations == 1

    assert affected.masks.cost is False
    assert affected.masks.latency is False


def test_nonterminal_records_excluded(tmp_path):
    records = make_records()

    records[0]["attempt_status"] = "uncertain"
    records[0]["strict_success"] = None
    records[0]["verification_status"] = None
    records[0]["cost_usd"] = None
    records[0]["latency_ms"] = None

    path = write_records(
        tmp_path,
        records,
    )

    loaded = load_collection_records([path])

    examples, statistics = aggregate_executions(loaded)

    assert statistics["excluded_nonterminal"] == 1

    affected = next(
        example
        for example in examples
        if (
            example.metadata.task_id == "task_00"
            and example.features.effort == "medium"
        )
    )

    assert affected.targets.trials == 1


def test_duplicate_execution_rejected(tmp_path):
    records = make_records()

    records.append(records[0].copy())

    path = write_records(
        tmp_path,
        records,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate execution record",
    ):
        load_collection_records([path])


def test_inconsistent_task_rejected(tmp_path):
    records = make_records()

    # Same task ID, different prompt.
    records[1]["prompt"] = "A different question."

    path = write_records(
        tmp_path,
        records,
    )

    with pytest.raises(
        ValueError,
        match="Inconsistent task definition",
    ):
        load_collection_records([path])


def test_invalid_success_label_rejected(tmp_path):
    records = make_records()

    records[0]["strict_success"] = True
    records[0]["verification_status"] = "incorrect"

    path = write_records(
        tmp_path,
        records,
    )

    with pytest.raises(
        ValueError,
        match="Invalid record",
    ):
        load_collection_records([path])


def test_smoke_excluded_by_default(tmp_path):
    records = make_records()

    for record in records:
        record["split"] = "smoke"

    path = write_records(
        tmp_path,
        records,
    )

    loaded = load_collection_records([path])

    examples, statistics = aggregate_executions(
        loaded,
    )

    assert examples == []
    assert statistics["excluded_smoke"] == 48

    examples, _ = aggregate_executions(
        loaded,
        include_smoke=True,
    )

    assert len(examples) == 24


def test_deterministic_splits(tmp_path):
    records = make_records()

    path = write_records(
        tmp_path,
        records,
    )

    output_a = tmp_path / "output_a"
    output_b = tmp_path / "output_b"

    build_ml_dataset(
        input_paths=[path],
        output_dir=output_a,
        validation_fraction=0.2,
        seed=42,
    )

    build_ml_dataset(
        input_paths=[path],
        output_dir=output_b,
        validation_fraction=0.2,
        seed=42,
    )

    for split in ["train", "validation", "test"]:
        assert (output_a / f"{split}.jsonl").read_bytes() == (
            output_b / f"{split}.jsonl"
        ).read_bytes()


def test_no_reference_answer_in_features(tmp_path):
    prepare_test_dataset(tmp_path)

    for split in ["train", "validation", "test"]:
        rows = read_ml_split(
            tmp_path / "ml",
            split,
        )

        for row in rows:
            features = row["features"]

            assert "reference_answer" not in features
            assert "strict_success" not in features
            assert "cost_usd" not in features
            assert "latency_ms" not in features

            assert "prompt" in features
            assert "model_id" in features
            assert "effort" in features


def test_manifest(tmp_path):
    manifest = prepare_test_dataset(tmp_path)

    assert manifest["schema_version"] == 1
    assert manifest["seed"] == 42

    assert manifest["statistics"]["input_records"] == 48

    assert manifest["statistics"]["aggregated_examples"] == 24

    assert manifest["splits"]["test"]["tasks"] == 2

    assert (tmp_path / "ml" / "manifest.json").is_file()
