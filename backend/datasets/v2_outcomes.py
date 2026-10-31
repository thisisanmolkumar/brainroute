from backend.datasets.v2_schema import TaskV2
from backend.datasets.v2_verifiers import (
    verify_task_response,
)


def classify_observation(
    task: TaskV2,
    *,
    provider_status: str,
    response_text: str | None,
) -> dict:

    if provider_status != "completed":
        return {
            "outcome": "execution_unverified",
            "success": None,
            "reason": provider_status,
        }

    if response_text is None:
        return {
            "outcome": "execution_unverified",
            "success": None,
            "reason": "missing_response_text",
        }

    try:
        result = verify_task_response(
            task,
            response_text,
        )

    except ValueError:
        return {
            "outcome": "verifier_error",
            "success": None,
            "reason": "verification_unavailable",
        }

    if result.correct:
        outcome = "correct"
    elif result.reason.startswith("missing_or_malformed"):
        outcome = "format_failure"
    else:
        outcome = "incorrect"

    return {
        "outcome": outcome,
        "success": result.correct,
        "reason": result.reason,
        "extracted_answer": result.extracted_answer,
    }
