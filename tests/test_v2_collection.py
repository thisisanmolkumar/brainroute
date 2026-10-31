import pytest

from backend.datasets.v2_collection import (
    build_collection_plan,
    overlaps_legacy,
)
from backend.datasets.v2_outcomes import (
    classify_observation,
)
from backend.datasets.v2_schema import TaskV2

CANDIDATES = [
    "gpt-6-luna:medium",
    "gpt-6-luna:high",
    "gpt-5.4-mini:medium",
    "gpt-5.4-mini:high",
]


def math_task(
    task_id: str,
    prompt: str,
    **overrides,
) -> TaskV2:
    values = {
        "task_id": task_id,
        "source": "gsm8k",
        "source_revision": "fixture-revision",
        "source_split": "train",
        "domain": "math",
        "raw_prompt": prompt,
        "reference_answer": "408",
        "verifier": "gsm8k_final_number",
        "verification_status": "ready",
        "family_id": task_id,
    }
    values.update(overrides)
    return TaskV2(**values)


def make_plan(tasks, **overrides):
    arguments = {
        "candidates": CANDIDATES,
        "legacy_prompts": [],
        "max_tasks": 8,
        "repeats": 1,
        "budget_usd": "0.50",
        "per_attempt_cap_usd": "0.005",
        "seed": 42,
    }
    arguments.update(overrides)

    return build_collection_plan(
        tasks,
        **arguments,
    )


def test_paired_candidate_coverage():
    tasks = [
        math_task("a", "Calculate 17 times 24."),
        math_task("b", "Find the product of 17 and 24."),
    ]

    plan = make_plan(tasks, repeats=2)

    assert plan["selected_tasks"] == 2
    assert plan["planned_attempts"] == 16

    assert len({item["attempt_id"] for item in plan["attempts"]}) == 16


def test_deterministic_plan():
    tasks = [
        math_task("a", "Question A about arithmetic."),
        math_task("b", "Question B about arithmetic."),
    ]

    first = make_plan(tasks)
    second = make_plan(list(reversed(tasks)))

    assert first["plan_id"] == second["plan_id"]
    assert first["attempts"] == second["attempts"]


def test_official_test_tasks_excluded():
    tasks = [
        math_task(
            "test-only",
            "Calculate 17 times 24.",
            source_split="test",
        ),
        math_task(
            "training",
            "Calculate 19 times 23.",
        ),
    ]

    plan = make_plan(tasks)

    assert plan["selected_tasks"] == 1
    assert plan["excluded"]["official_nontraining_split"] == 1


def test_pending_tasks_excluded():
    tasks = [
        math_task(
            "pending",
            "Unverified question.",
            verification_status="pending",
            verifier=None,
        ),
        math_task(
            "ready",
            "Calculate 17 times 24.",
        ),
    ]

    plan = make_plan(tasks)

    assert plan["selected_tasks"] == 1
    assert plan["excluded"]["verifier_not_ready"] == 1


def test_legacy_wrapped_prompt_detected():
    prompt = "What is 17 multiplied by 24?"

    old = "Solve the mathematical problem.\nFINAL_ANSWER: <number>\n" + prompt

    assert overlaps_legacy(
        prompt,
        [old],
    )


def test_legacy_overlap_excluded():
    old_prompt = "What is 17 multiplied by 24?"

    tasks = [
        math_task("old", old_prompt),
        math_task("new", "Calculate 19 times 23."),
    ]

    plan = make_plan(
        tasks,
        legacy_prompts=[old_prompt],
    )

    assert plan["selected_tasks"] == 1
    assert plan["excluded"]["dataset_v1_overlap"] == 1


def test_family_variant_not_repeated():
    tasks = [
        math_task(
            "variant-a",
            "Calculate 17 times 24.",
            family_id="same-family",
        ),
        math_task(
            "variant-b",
            "Find the product of 17 and 24.",
            family_id="same-family",
        ),
    ]

    plan = make_plan(tasks)

    assert plan["selected_tasks"] == 1


def test_reference_not_appended_to_prompt():
    task = math_task(
        "a",
        "Calculate 17 times 24.",
    )

    plan = make_plan([task])

    for item in plan["attempts"]:
        assert "FINAL_ANSWER: 408" not in (item["execution_prompt"])

        assert item["execution_prompt"].startswith(task.raw_prompt)


def test_budget_validation():
    with pytest.raises(
        ValueError,
        match="exceeds total budget",
    ):
        make_plan(
            [math_task("a", "Calculate 17 times 24.")],
            budget_usd="0.01",
            per_attempt_cap_usd="0.02",
        )


def test_correct_observation():
    task = math_task(
        "a",
        "Calculate 17 times 24.",
    )

    result = classify_observation(
        task,
        provider_status="completed",
        response_text="FINAL_ANSWER: 408",
    )

    assert result["outcome"] == "correct"
    assert result["success"] is True


def test_format_failure_observation():
    task = math_task(
        "a",
        "Calculate 17 times 24.",
    )

    result = classify_observation(
        task,
        provider_status="completed",
        response_text="The answer is 408.",
    )

    assert result["outcome"] == "format_failure"
    assert result["success"] is False


def test_timeout_is_not_task_failure():
    task = math_task(
        "a",
        "Calculate 17 times 24.",
    )

    result = classify_observation(
        task,
        provider_status="timeout",
        response_text=None,
    )

    assert result["outcome"] == ("execution_unverified")
    assert result["success"] is None
