from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.benchmarks.schema import VerifierType

VerificationStatus = Literal[
    "correct",
    "incorrect",
    "unparseable",
    "skipped",
]


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str

    verifier: VerifierType

    status: VerificationStatus

    verified: bool
    correct: bool | None

    predicted_answer: str | None = None
    reference_answer: str

    reason: str | None = None
