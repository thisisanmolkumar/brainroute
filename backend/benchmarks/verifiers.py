import re
from decimal import Decimal
from typing import Literal

from backend.benchmarks.schema import BenchmarkTask
from backend.benchmarks.verification import (
    VerificationResult,
    VerificationStatus,
)

# ----------------------------------------
# Exceptions
# ----------------------------------------


class AnswerFormatError(ValueError):
    """The model's answer cannot be reliably parsed."""


# ----------------------------------------
# Regular Expressions
# ----------------------------------------

FINAL_ANSWER_PATTERN = re.compile(
    r"FINAL_ANSWER\s*:\s*(.*)",
    flags=re.IGNORECASE,
)

NUMERIC_PATTERN = re.compile(
    r"[+-]?"
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)


# ----------------------------------------
# Final Answer Extraction
# ----------------------------------------


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from an LLM response.

    Rules:
    - Exactly one FINAL_ANSWER line must exist.
    - It must be the last nonempty line.
    - The answer must not be empty.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    matches = []

    for index, line in enumerate(lines):
        match = FINAL_ANSWER_PATTERN.fullmatch(line)

        if match is not None:
            matches.append((index, match.group(1).strip()))

    if not matches:
        raise AnswerFormatError("Missing FINAL_ANSWER line")

    if len(matches) > 1:
        raise AnswerFormatError("Multiple FINAL_ANSWER lines found")

    index, answer = matches[0]

    if index != len(lines) - 1:
        raise AnswerFormatError("FINAL_ANSWER must be the last nonempty line")

    if not answer:
        raise AnswerFormatError("FINAL_ANSWER cannot be empty")

    return answer


# ----------------------------------------
# Numeric Answer Parsing
# ----------------------------------------


def parse_numeric_answer(answer: str) -> Decimal:
    """
    Parse a numeric answer using Decimal.

    Supports:
    - Integers
    - Decimals
    - Negative numbers
    - Properly grouped thousands separators
    - Scientific notation

    Rejects:
    - Units
    - Percent signs
    - Invalid comma grouping
    - NaN and Infinity
    """

    answer = answer.strip()

    if not NUMERIC_PATTERN.fullmatch(answer):
        raise ValueError(f"Invalid numeric answer: {answer!r}")

    normalized = answer.replace(",", "")

    value = Decimal(normalized)

    if not value.is_finite():
        raise ValueError("Numeric answer must be finite")

    return value


# ----------------------------------------
# Verification Result Helper
# ----------------------------------------


def build_result(
    task: BenchmarkTask,
    status: VerificationStatus,
    predicted_answer: str | None = None,
    reason: str | None = None,
) -> VerificationResult:

    verified = status in {
        "correct",
        "incorrect",
    }

    correct = status == "correct" if verified else None

    return VerificationResult(
        task_id=task.task_id,
        verifier=task.verifier,
        status=status,
        verified=verified,
        correct=correct,
        predicted_answer=predicted_answer,
        reference_answer=task.reference_answer,
        reason=reason,
    )


# ----------------------------------------
# Verification Service
# ----------------------------------------


class VerificationService:
    def verify(
        self,
        task: BenchmarkTask,
        output_text: str,
        execution_status: str = "completed",
    ) -> VerificationResult:

        # ----------------------------------
        # 1. Check execution status
        # ----------------------------------

        if execution_status != "completed":
            return build_result(
                task=task,
                status="skipped",
                reason=(f"Execution status: {execution_status}"),
            )

        # ----------------------------------
        # 2. Validate reference answer
        # ----------------------------------

        if task.verifier == "numeric_final":
            try:
                reference = parse_numeric_answer(task.reference_answer)

            except ValueError as exc:
                raise ValueError(
                    f"Invalid reference answer for task '{task.task_id}'"
                ) from exc

        elif task.verifier == "exact_final":
            reference = task.reference_answer.strip().casefold()

        else:
            raise ValueError(f"Unsupported verifier: {task.verifier}")

        # ----------------------------------
        # 3. Extract model's final answer
        # ----------------------------------

        try:
            predicted_answer = extract_final_answer(output_text)

        except AnswerFormatError as exc:
            return build_result(
                task=task,
                status="unparseable",
                reason=str(exc),
            )

        # ----------------------------------
        # 4. Evaluate numeric answers
        # ----------------------------------

        if task.verifier == "numeric_final":
            try:
                predicted = parse_numeric_answer(predicted_answer)

            except ValueError as exc:
                return build_result(
                    task=task,
                    status="unparseable",
                    predicted_answer=predicted_answer,
                    reason=str(exc),
                )

            is_correct = predicted == reference

        # ----------------------------------
        # 5. Evaluate exact answers
        # ----------------------------------

        else:
            predicted = predicted_answer.strip().casefold()

            is_correct = predicted == reference

        # ----------------------------------
        # 6. Return verification result
        # ----------------------------------

        status: Literal["correct", "incorrect"] = (
            "correct" if is_correct else "incorrect"
        )

        return build_result(
            task=task,
            status=status,
            predicted_answer=predicted_answer,
            reason=(None if is_correct else "Final answer differs from reference"),
        )
