import pytest

from backend.datasets.v2_adapters import (
    adapt_bbh,
    adapt_gsm8k,
    adapt_ifeval,
    adapt_mbpp,
    adapt_record,
    gsm8k_final_answer,
)

REVISION = "test-commit-123"


def test_gsm8k_final_answer():
    answer = "First calculate 20 + 22 = 42.\nThe final answer is 42.\n#### 42"

    assert gsm8k_final_answer(answer) == "42"

    assert gsm8k_final_answer("Calculate the total.\n#### 1,234") == "1234"


def test_gsm8k_adapter():
    record = {
        "question": "What is 17 multiplied by 24?",
        "answer": "17 * 24 = 408.\n#### 408",
    }

    task = adapt_gsm8k(
        record,
        revision=REVISION,
        split="train",
        index=0,
    )

    assert task.domain == "math"
    assert task.raw_prompt == record["question"]
    assert task.reference_answer == "408"
    assert task.verifier == "gsm8k_final_number"
    assert task.verification_status == "ready"


def test_gsm8k_rejects_missing_answer():
    with pytest.raises(
        ValueError,
        match="missing its final",
    ):
        gsm8k_final_answer("The answer is probably 42.")


def test_mbpp_sanitized_adapter():
    record = {
        "task_id": 12,
        "prompt": "Write a function to add two integers.",
        "code": "def add(a, b): return a + b",
        "test_imports": [],
        "test_list": [
            "assert add(1, 2) == 3",
            "assert add(-1, 1) == 0",
        ],
    }

    task = adapt_mbpp(
        record,
        revision=REVISION,
        split="train",
        config="sanitized",
    )

    assert task.domain == "coding"
    assert task.source_split == "train"
    assert task.verification_status == "pending"
    assert task.reference_answer is None
    assert task.metadata["requires_isolated_execution"] is True

    assert "def add" not in task.raw_prompt


def test_mbpp_full_adapter():
    record = {
        "task_id": 12,
        "text": "Write a function to add two integers.",
        "code": "def add(a, b): return a + b",
        "test_setup_code": "",
        "test_list": [
            "assert add(1, 2) == 3",
        ],
    }

    task = adapt_mbpp(
        record,
        revision=REVISION,
        split="test",
        config="full",
    )

    assert task.source_split == "test"
    assert task.metadata["test_list"]


def test_mbpp_prompt_partition_rejected():
    with pytest.raises(
        ValueError,
        match="few-shot prompt",
    ):
        adapt_mbpp(
            {},
            revision=REVISION,
            split="prompt",
            config="sanitized",
        )


def test_bbh_boolean_adapter():
    task = adapt_bbh(
        {
            "input": "True and False is",
            "target": "False",
        },
        revision=REVISION,
        split="test",
        config="boolean_expressions",
        index=2,
    )

    assert task.source_split == "test"
    assert task.verification_status == "ready"
    assert task.verifier == "bbh_boolean"


def test_bbh_other_tasks_pending():
    task = adapt_bbh(
        {
            "input": "A complex logical deduction...",
            "target": "(B)",
        },
        revision=REVISION,
        split="test",
        config="logical_deduction_three_objects",
        index=3,
    )

    assert task.verification_status == "pending"
    assert task.verifier is None


def test_bbh_cannot_enter_training():
    with pytest.raises(
        ValueError,
        match="official test",
    ):
        adapt_bbh(
            {
                "input": "True and False is",
                "target": "False",
            },
            revision=REVISION,
            split="train",
            config="boolean_expressions",
            index=0,
        )


def test_ifeval_adapter():
    task = adapt_ifeval(
        {
            "key": 1000,
            "prompt": "Write three paragraphs without commas.",
            "instruction_id_list": [
                "punctuation:no_comma",
            ],
            "kwargs": [{}],
        },
        revision=REVISION,
        split="train",
    )

    assert task.domain == "instruction"
    assert task.source_split == "test"
    assert task.verification_status == "pending"

    assert task.metadata["instruction_ids"] == ["punctuation:no_comma"]


def test_ifeval_invalid_metadata_rejected():
    with pytest.raises(
        ValueError,
        match="Invalid IFEval",
    ):
        adapt_ifeval(
            {
                "key": 1,
                "prompt": "Answer without commas.",
                "instruction_id_list": [
                    "punctuation:no_comma",
                ],
                "kwargs": [],
            },
            revision=REVISION,
            split="train",
        )


def test_unknown_dataset_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported dataset",
    ):
        adapt_record(
            "unknown",
            {},
            revision=REVISION,
            split="train",
            index=0,
        )


def test_source_revision_required():
    with pytest.raises(
        ValueError,
        match="revision is required",
    ):
        adapt_record(
            "gsm8k",
            {
                "question": "Calculate 2 + 2.",
                "answer": "#### 4",
            },
            revision="",
            split="train",
            index=0,
        )
