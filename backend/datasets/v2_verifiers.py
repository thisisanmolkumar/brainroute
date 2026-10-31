import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from backend.datasets.v2_schema import TaskV2

NUMBER_PATTERN = (
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?"
)

NUMERIC_OUTPUT_RE = re.compile(
    rf"FINAL_ANSWER:\s*({NUMBER_PATTERN})",
    re.IGNORECASE,
)

BOOLEAN_OUTPUT_RE = re.compile(
    r"FINAL_ANSWER:\s*(True|False)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    reason: str
    extracted_answer: str | None = None


def final_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return lines[-1] if lines else ""


def verify_final_number(
    response: str,
    reference: str,
) -> VerificationResult:

    last_line = final_nonempty_line(response)

    match = NUMERIC_OUTPUT_RE.fullmatch(last_line)

    if match is None:
        return VerificationResult(
            correct=False,
            reason="missing_or_malformed_final_answer",
        )

    try:
        predicted = Decimal(match.group(1).replace(",", ""))

        expected = Decimal(reference)

    except InvalidOperation:
        return VerificationResult(
            correct=False,
            reason="invalid_numeric_answer",
        )

    if not predicted.is_finite() or not expected.is_finite():
        return VerificationResult(
            correct=False,
            reason="nonfinite_numeric_answer",
        )

    correct = predicted == expected

    return VerificationResult(
        correct=correct,
        reason="correct" if correct else "incorrect",
        extracted_answer=format(predicted, "f"),
    )


def verify_boolean(
    response: str,
    reference: str,
) -> VerificationResult:

    last_line = final_nonempty_line(response)

    match = BOOLEAN_OUTPUT_RE.fullmatch(last_line)

    if match is None:
        return VerificationResult(
            correct=False,
            reason="missing_or_malformed_boolean_answer",
        )

    predicted = match.group(1).lower()
    expected = reference.lower()

    if expected not in {"true", "false"}:
        raise ValueError("Invalid Boolean reference answer")

    return VerificationResult(
        correct=predicted == expected,
        reason=("correct" if predicted == expected else "incorrect"),
        extracted_answer=predicted,
    )


VERIFIERS: dict[
    str,
    Callable[[str, str], VerificationResult],
] = {
    "gsm8k_final_number": verify_final_number,
    "bbh_boolean": verify_boolean,
}


def verify_task_response(
    task: TaskV2,
    response: str,
) -> VerificationResult:

    if task.verification_status != "ready":
        raise ValueError(f"Task {task.task_id} is not verifier-ready")

    if task.reference_answer is None:
        raise ValueError("Task lacks a reference answer")

    verifier = VERIFIERS.get(task.verifier or "")

    if verifier is None:
        raise ValueError(f"Unregistered verifier: {task.verifier}")

    return verifier(
        response,
        task.reference_answer,
    )
