import json

import pytest
from pydantic import ValidationError

from backend.datasets.v2_builder import (
    assigned_split,
    build_groups,
    compile_dataset,
    content_hash,
    normalize_prompt,
)
from backend.datasets.v2_schema import TaskV2


def make_task(
    task_id="task-001",
    prompt="What is 17 multiplied by 24?",
    **overrides,
):
    data = {
        "task_id": task_id,
        "source": "brainroute-original",
        "source_revision": "v1",
        "source_split": "train",
        "domain": "math",
        "difficulty": "easy",
        "raw_prompt": prompt,
        "reference_answer": "408",
        "family_id": task_id,
        "verification_status": "pending",
    }

    data.update(overrides)

    return TaskV2.model_validate(data)


def write_tasks(path, tasks):
    path.write_text(
        "\n".join(task.model_dump_json() for task in tasks) + "\n",
        encoding="utf-8",
    )


def test_prompt_normalization():
    first = "  What is 17   multiplied by 24?  "
    second = "what is 17 multiplied by 24?"

    assert normalize_prompt(first) == (normalize_prompt(second))

    assert content_hash(first) == content_hash(second)


def test_raw_prompt_is_preserved():
    prompt = "  What is 17 multiplied by 24?  "

    task = make_task(prompt=prompt)

    assert task.raw_prompt == prompt


def test_ready_task_needs_verifier():
    with pytest.raises(
        ValidationError,
        match="Ready tasks require",
    ):
        make_task(
            verification_status="ready",
            verifier=None,
        )


def test_family_variants_stay_grouped():
    tasks = [
        make_task(
            "variant-a",
            "Calculate 17 times 24.",
            family_id="same-problem",
        ),
        make_task(
            "variant-b",
            "What is the product of 17 and 24?",
            family_id="same-problem",
        ),
    ]

    groups = build_groups(tasks)

    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_exact_duplicates_stay_grouped():
    tasks = [
        make_task(
            "task-a",
            "What is 17 multiplied by 24?",
        ),
        make_task(
            "task-b",
            "  what is 17 multiplied by 24?  ",
        ),
    ]

    groups = build_groups(tasks)

    assert len(groups) == 1


def test_official_test_remains_test():
    task = make_task(source_split="test")

    assert (
        assigned_split(
            [task],
            {},
            seed=42,
        )
        == "test"
    )


def test_historical_assignment_preserved():
    task = make_task()

    locked = {
        (
            task.source,
            task.task_id,
        ): "validation",
    }

    assert (
        assigned_split(
            [task],
            locked,
            seed=42,
        )
        == "validation"
    )


def test_cross_split_family_rejected():
    tasks = [
        make_task(
            "task-a",
            "Question A",
            family_id="shared",
        ),
        make_task(
            "task-b",
            "Question B",
            family_id="shared",
            source_split="test",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="cross official source splits",
    ):
        assigned_split(
            tasks,
            {},
            seed=42,
        )


def test_dataset_compilation(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "compiled"

    tasks = [
        make_task(
            f"task-{index}",
            f"Calculate {index} plus {index + 3}.",
            reference_answer=str(2 * index + 3),
        )
        for index in range(50)
    ]

    write_tasks(source, tasks)

    manifest = compile_dataset(
        [source],
        output,
        seed=42,
    )

    assert manifest["source_tasks"] == 50
    assert manifest["compiled_tasks"] == 50

    assert sum(data["tasks"] for data in manifest["splits"].values()) == 50

    assert (output / "manifest.json").is_file()

    for split in [
        "train",
        "validation",
        "calibration",
        "test",
    ]:
        assert (output / f"pool_{split}.jsonl").is_file()


def test_compilation_is_deterministic(tmp_path):
    source = tmp_path / "source.jsonl"

    write_tasks(
        source,
        [
            make_task(
                f"task-{index}",
                f"Question number {index}",
            )
            for index in range(30)
        ],
    )

    output_a = tmp_path / "a"
    output_b = tmp_path / "b"

    first = compile_dataset(
        [source],
        output_a,
        seed=42,
    )

    second = compile_dataset(
        [source],
        output_b,
        seed=42,
    )

    assert first["splits"] == second["splits"]

    for split in first["splits"]:
        assert (output_a / f"pool_{split}.jsonl").read_bytes() == (
            output_b / f"pool_{split}.jsonl"
        ).read_bytes()


def test_conflicting_duplicate_answers(tmp_path):
    source = tmp_path / "source.jsonl"

    write_tasks(
        source,
        [
            make_task("task-a"),
            make_task(
                "task-b",
                reference_answer="999",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Conflicting reference answers",
    ):
        compile_dataset(
            [source],
            tmp_path / "output",
        )


def test_compiled_manifest_is_json(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "compiled"

    write_tasks(source, [make_task()])

    compile_dataset([source], output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 2
    assert manifest["dataset_name"] == ("brainroute_dataset_v2")
