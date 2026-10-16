from pathlib import Path

import pytest

from backend.benchmarks.loader import load_tasks

SAMPLE_DATASET = (
    Path(__file__).resolve().parents[1] / "datasets" / "samples" / "smoke.jsonl"
)


def test_load_tasks():
    tasks = load_tasks(SAMPLE_DATASET)

    assert len(tasks) == 3
    assert tasks[0].task_id == "smoke_math_001"


def test_task_schema():
    tasks = load_tasks(SAMPLE_DATASET)

    task = tasks[0]

    assert task.domain == "math"
    assert task.reference_answer == "408"
    assert task.verifier == "numeric_final"


def test_split_filter():
    tasks = load_tasks(
        SAMPLE_DATASET,
        split="smoke",
    )

    assert len(tasks) == 3

    tasks = load_tasks(
        SAMPLE_DATASET,
        split="train",
    )

    assert len(tasks) == 0


def test_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_tasks("nonexistent.jsonl")


def test_invalid_task(tmp_path):
    path = tmp_path / "invalid.jsonl"

    path.write_text(
        '{"task_id":"invalid"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid benchmark task"):
        load_tasks(path)


def test_duplicate_task_ids(tmp_path):
    original = SAMPLE_DATASET.read_text(encoding="utf-8")
    first_line = original.splitlines()[0]

    path = tmp_path / "duplicates.jsonl"

    path.write_text(
        first_line + "\n" + first_line + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate task ID"):
        load_tasks(path)


def test_ground_truth_is_separate():
    tasks = load_tasks(SAMPLE_DATASET)

    task = tasks[0]

    # Ground truth is not included in the
    # serialized prompt field.
    assert "408" not in task.prompt

    assert task.reference_answer == "408"
