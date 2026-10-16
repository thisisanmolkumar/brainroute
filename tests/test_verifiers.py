import pytest

from backend.benchmarks.loader import load_tasks
from backend.benchmarks.verifiers import (
    AnswerFormatError,
    VerificationService,
    extract_final_answer,
    parse_numeric_answer,
)


@pytest.fixture
def tasks():
    return load_tasks("datasets/samples/smoke.jsonl")


@pytest.fixture
def verifier():
    return VerificationService()


# ----------------------------------------
# Final Answer Extraction
# ----------------------------------------


def test_extract_final_answer():

    text = """
    17 multiplied by 24 is 408.

    FINAL_ANSWER: 408
    """

    assert extract_final_answer(text) == "408"


def test_missing_final_answer():

    with pytest.raises(
        AnswerFormatError,
        match="Missing FINAL_ANSWER",
    ):
        extract_final_answer("The answer is 408.")


def test_multiple_final_answers():

    text = """
    FINAL_ANSWER: 400
    FINAL_ANSWER: 408
    """

    with pytest.raises(
        AnswerFormatError,
        match="Multiple",
    ):
        extract_final_answer(text)


def test_final_answer_not_last():

    text = """
    FINAL_ANSWER: 408
    I think this is correct.
    """

    with pytest.raises(
        AnswerFormatError,
        match="last nonempty line",
    ):
        extract_final_answer(text)


def test_empty_final_answer():

    with pytest.raises(
        AnswerFormatError,
        match="cannot be empty",
    ):
        extract_final_answer("FINAL_ANSWER:")


# ----------------------------------------
# Numeric Parsing
# ----------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "408",
        "408.0",
        "408.00",
        "+408",
        "4.08e2",
    ],
)
def test_numeric_equivalence(answer):

    assert parse_numeric_answer(answer) == 408


@pytest.mark.parametrize(
    "answer",
    [
        "408 km",
        "1,00",
        "banana",
        "NaN",
        "Infinity",
        "50%",
    ],
)
def test_invalid_numeric_values(answer):

    with pytest.raises(ValueError):
        parse_numeric_answer(answer)


def test_thousands_separator():

    assert parse_numeric_answer("1,234.50") == (parse_numeric_answer("1234.5"))


def test_negative_number():

    assert parse_numeric_answer("-12.50") == (parse_numeric_answer("-12.5"))


# ----------------------------------------
# Numeric Verification
# ----------------------------------------


def test_correct_numeric_answer(tasks, verifier):

    task = tasks[0]

    result = verifier.verify(
        task,
        "17 * 24 = 408\nFINAL_ANSWER: 408",
    )

    assert result.status == "correct"
    assert result.verified is True
    assert result.correct is True


def test_equivalent_numeric_answer(tasks, verifier):

    result = verifier.verify(
        tasks[0],
        "FINAL_ANSWER: 408.00",
    )

    assert result.correct is True


def test_incorrect_numeric_answer(tasks, verifier):

    result = verifier.verify(
        tasks[0],
        "FINAL_ANSWER: 409",
    )

    assert result.status == "incorrect"
    assert result.verified is True
    assert result.correct is False


def test_unparseable_numeric_answer(tasks, verifier):

    result = verifier.verify(
        tasks[0],
        "FINAL_ANSWER: banana",
    )

    assert result.status == "unparseable"
    assert result.verified is False
    assert result.correct is None


# ----------------------------------------
# Exact Match Verification
# ----------------------------------------


def test_correct_exact_answer(tasks, verifier):

    task = tasks[2]

    result = verifier.verify(
        task,
        "FINAL_ANSWER: YES",
    )

    assert result.status == "correct"
    assert result.correct is True


def test_incorrect_exact_answer(tasks, verifier):

    result = verifier.verify(
        tasks[2],
        "FINAL_ANSWER: no",
    )

    assert result.status == "incorrect"
    assert result.correct is False


def test_no_partial_matching(tasks, verifier):

    result = verifier.verify(
        tasks[2],
        "FINAL_ANSWER: yes or no",
    )

    assert result.correct is False


# ----------------------------------------
# Execution Status Handling
# ----------------------------------------


def test_incomplete_execution(tasks, verifier):

    result = verifier.verify(
        tasks[0],
        "FINAL_ANSWER: 408",
        execution_status="incomplete",
    )

    assert result.status == "skipped"
    assert result.verified is False
    assert result.correct is None


def test_failed_execution(tasks, verifier):

    result = verifier.verify(
        tasks[0],
        "",
        execution_status="failed",
    )

    assert result.status == "skipped"
    assert result.correct is None


# ----------------------------------------
# Invalid Benchmark References
# ----------------------------------------


def test_invalid_reference_answer(tasks, verifier):

    task = tasks[0].model_copy(update={"reference_answer": "four hundred eight"})

    with pytest.raises(
        ValueError,
        match="Invalid reference answer",
    ):
        verifier.verify(
            task,
            "FINAL_ANSWER: 408",
        )


# ----------------------------------------
# Structured Output
# ----------------------------------------


def test_verification_result_serialization(
    tasks,
    verifier,
):

    result = verifier.verify(
        tasks[0],
        "FINAL_ANSWER: 408",
    )

    data = result.model_dump()

    assert data["task_id"] == "smoke_math_001"
    assert data["status"] == "correct"
    assert data["correct"] is True
    assert data["reference_answer"] == "408"
