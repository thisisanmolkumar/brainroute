from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

Domain = Literal[
    "math",
    "coding",
    "logic",
    "instruction",
    "knowledge",
]

Difficulty = Literal[
    "easy",
    "medium",
    "hard",
    "unknown",
]

SourceSplit = Literal[
    "train",
    "validation",
    "test",
]

VerificationStatus = Literal[
    "ready",
    "pending",
]


class TaskV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)

    source_split: SourceSplit

    domain: Domain
    difficulty: Difficulty = "unknown"

    raw_prompt: str = Field(min_length=1)

    reference_answer: str | None = None
    verifier: str | None = None

    family_id: str | None = None

    verification_status: VerificationStatus = "pending"

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_task(self):
        if not self.raw_prompt.strip():
            raise ValueError("Empty task prompt")

        if self.verification_status == "ready" and (
            not self.verifier or self.reference_answer is None
        ):
            raise ValueError("Ready tasks require a verifier and reference answer")

        if self.family_id is None:
            self.family_id = self.task_id

        return self
