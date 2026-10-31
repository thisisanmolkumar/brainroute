import pytest

from backend.datasets.v2_adapters import (
    adapt_bbh,
    adapt_gsm8k,
    adapt_mbpp,
)
from backend.datasets.v2_readiness import (
    assert_collectible,
    render_execution_prompt,
)
from backend.datasets.v2_verifiers import (
    verify_task_response,
)


def math_task():
    return adapt_gsm8k(
        {
            "question": "Calculate 17 multiplied by 24.",
            "answer": "17 * 24 = 408.\n#### 408",
        },
        revision="fixture-v1",
        split="train",
        index=0,
    )


def boolean_task():
    return adapt_bbh(
        {
            "input": "True and False is",
            "target": "False",
        },
        revision="fixture-v1",
        split="test",
        config="boolean_expressions",
        index=0,
    )


def test_math_correct_answer():
    result = verify_task_response(
        math_task(),
        "17 * 24 = 408\nFINAL_ANSWER: 408",
    )

    assert result.correct is True
    assert result.extracted_answer == "408"


def test_math_incorrect_answer():
    result = verify_task_response(
        math_task(),
        "FINAL_ANSWER: 407",
    )

    assert result.correct is False
    assert result.reason == "incorrect"


def test_math_equivalent_decimal():
    result = verify_task_response(
        math_task(),
        "FINAL_ANSWER: 408.0",
    )

    assert result.correct is True


def test_math_format_failure():
    result = verify_task_response(
        math_task(),
        "I think the answer is 408.",
    )

    assert result.correct is False
    assert result.reason == ("missing_or_malformed_final_answer")


def test_boolean_correct():
    result = verify_task_response(
        boolean_task(),
        "FINAL_ANSWER: False",
    )

    assert result.correct is True


def test_boolean_incorrect():
    result = verify_task_response(
        boolean_task(),
        "FINAL_ANSWER: True",
    )

    assert result.correct is False


def test_pending_coding_task_rejected():
    task = adapt_mbpp(
        {
            "task_id": 12,
            "prompt": "Write an add function.",
            "code": "def add(a, b): return a + b",
            "test_imports": [],
            "test_list": [
                "assert add(1, 2) == 3",
            ],
        },
        revision="fixture-v1",
        split="train",
        config="sanitized",
    )

    with pytest.raises(
        ValueError,
        match="verification is pending",
    ):
        assert_collectible(task)

    with pytest.raises(
        ValueError,
        match="not verifier-ready",
    ):
        verify_task_response(
            task,
            "def add(a, b): return a + b",
        )


def test_execution_prompt_separated_from_raw():
    task = math_task()

    original = task.raw_prompt
    execution_prompt = render_execution_prompt(task)

    assert task.raw_prompt == original
    assert execution_prompt.startswith(original)
    assert "FINAL_ANSWER:" in execution_prompt

    # The reference answer must not be appended
    # as a separate answer key.
    assert "FINAL_ANSWER: 408" not in execution_prompt
