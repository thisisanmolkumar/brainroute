from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskDomain = Literal[
    "math",
    "logic",
    "instruction",
    "code",
]

DatasetSplit = Literal[
    "train",
    "validation",
    "test",
    "smoke",
]

VerifierType = Literal[
    "numeric_final",
    "exact_final",
]


class BenchmarkTask(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    # Unique identifier for this benchmark problem
    task_id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_.:-]+$",
    )

    # Task information
    domain: TaskDomain
    source: str = Field(min_length=1)
    split: DatasetSplit

    # The ONLY task text sent to the LLM
    prompt: str = Field(min_length=1)

    # Ground truth, never sent to the LLM
    reference_answer: str = Field(min_length=1)

    # Verification strategy
    verifier: VerifierType

    # Optional family identifier for preventing
    # related problems from crossing data splits
    family_id: str | None = None

    # Additional dataset-specific information
    metadata: dict[
        str,
        str | int | float | bool | None,
    ] = Field(default_factory=dict)
