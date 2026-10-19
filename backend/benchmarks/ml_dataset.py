import hashlib
import json
import random
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

TERMINAL_STATUSES = {"completed", "error"}

DatasetSplit = Literal[
    "train",
    "validation",
    "test",
    "smoke",
]

MLSplit = Literal[
    "train",
    "validation",
    "test",
]


class CollectionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: int
    task_id: str
    source: str
    domain: str
    split: DatasetSplit

    family_id: str | None = None

    prompt: str = Field(min_length=1)
    reference_answer: str

    model_id: str
    effort: str
    repeat_index: int = Field(ge=0)

    attempt_status: Literal[
        "pending",
        "running",
        "completed",
        "error",
        "uncertain",
    ]

    verification_status: (
        Literal[
            "correct",
            "incorrect",
            "unparseable",
            "skipped",
        ]
        | None
    ) = None
    strict_success: bool | None = None

    cost_usd: Decimal | None = Field(
        default=None,
        ge=0,
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_labels(self):
        terminal = self.attempt_status in TERMINAL_STATUSES

        if terminal and self.strict_success is None:
            raise ValueError("Terminal attempts require a success label")

        if not terminal and self.strict_success is not None:
            raise ValueError("Nonterminal attempts cannot have success labels")

        if self.strict_success is True and self.verification_status != "correct":
            raise ValueError("Successful attempt must have correct verification")

        if self.verification_status == "correct" and self.strict_success is not True:
            raise ValueError("Correct verification requires successful attempt")

        return self


class MLFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    model_id: str
    effort: str

    character_count: int
    word_count: int
    line_count: int

    digit_count: int
    question_mark_count: int
    operator_count: int

    has_code_block: bool


class MLTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success_rate: float = Field(ge=0, le=1)
    successes: int = Field(ge=0)
    trials: int = Field(gt=0)

    mean_cost_usd: Decimal | None = None
    mean_latency_ms: float | None = None

    cost_observations: int = Field(ge=0)
    latency_observations: int = Field(ge=0)


class TargetMasks(BaseModel):
    cost: bool
    latency: bool


class MLMetadata(BaseModel):
    source: str
    task_id: str
    domain: str
    family_id: str | None

    original_split: DatasetSplit
    group_key: str

    run_ids: list[int]


class MLExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: MLFeatures
    targets: MLTargets
    masks: TargetMasks
    metadata: MLMetadata


def load_collection_records(
    paths: list[str | Path],
) -> list[CollectionRecord]:

    if not paths:
        raise ValueError("No collection exports supplied")

    records = []
    seen_attempts = set()

    # Tracks consistency across different runs.
    task_definitions = {}
    prompt_owners = {}

    for source_path in paths:
        path = Path(source_path)

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                if not line.strip():
                    continue

                try:
                    record = CollectionRecord.model_validate_json(line)

                except ValidationError as exc:
                    raise ValueError(
                        f"Invalid record at {path}:{line_number}: {exc}"
                    ) from exc

                # ----------------------------------
                # 1. Validate attempt identity
                # ----------------------------------

                attempt_key = (
                    record.run_id,
                    record.source,
                    record.task_id,
                    record.model_id,
                    record.effort,
                    record.repeat_index,
                )

                if attempt_key in seen_attempts:
                    raise ValueError(f"Duplicate execution record: {attempt_key}")

                seen_attempts.add(attempt_key)

                # ----------------------------------
                # 2. Validate task consistency
                # ----------------------------------

                task_key = (
                    record.source,
                    record.task_id,
                )

                definition = (
                    record.prompt,
                    record.reference_answer,
                    record.domain,
                    record.split,
                    record.family_id,
                )

                previous = task_definitions.get(task_key)

                if previous is not None and previous != definition:
                    raise ValueError(f"Inconsistent task definition: {task_key}")

                task_definitions[task_key] = definition

                # ----------------------------------
                # 3. Reject exact prompt duplicates
                #    with different task identities
                # ----------------------------------

                normalized_prompt = " ".join(record.prompt.split())

                prompt_hash = hashlib.sha256(
                    normalized_prompt.encode("utf-8")
                ).hexdigest()

                previous_owner = prompt_owners.get(prompt_hash)

                if previous_owner is not None and previous_owner != task_key:
                    raise ValueError(
                        "Identical prompt belongs to different "
                        "task IDs. Deduplicate before splitting."
                    )

                prompt_owners[prompt_hash] = task_key

                records.append(record)

    if not records:
        raise ValueError("No collection records found")

    return records


def extract_features(
    prompt: str,
    model_id: str,
    effort: str,
) -> MLFeatures:

    words = re.findall(
        r"\b\w+\b",
        prompt,
    )

    operators = set("+-*/=<>%")

    return MLFeatures(
        prompt=prompt,
        model_id=model_id,
        effort=effort,
        character_count=len(prompt),
        word_count=len(words),
        line_count=len(prompt.splitlines()),
        digit_count=sum(character.isdigit() for character in prompt),
        question_mark_count=prompt.count("?"),
        operator_count=sum(character in operators for character in prompt),
        has_code_block="```" in prompt,
    )


def task_group_key(
    record: CollectionRecord,
) -> str:

    if record.family_id:
        return f"family:{record.source}:{record.family_id}"

    return f"task:{record.source}:{record.task_id}"


def aggregate_executions(
    records: list[CollectionRecord],
    include_smoke: bool = False,
) -> tuple[list[MLExample], dict]:

    grouped = defaultdict(list)

    excluded_smoke = 0
    excluded_nonterminal = 0

    for record in records:
        if record.split == "smoke" and not include_smoke:
            excluded_smoke += 1
            continue

        if record.attempt_status not in TERMINAL_STATUSES:
            excluded_nonterminal += 1
            continue

        key = (
            record.source,
            record.task_id,
            record.model_id,
            record.effort,
        )

        grouped[key].append(record)

    examples = []

    for key, group in sorted(grouped.items()):
        first = group[0]

        trials = len(group)

        successes = sum(record.strict_success is True for record in group)

        costs = [record.cost_usd for record in group if record.cost_usd is not None]

        latencies = [
            record.latency_ms for record in group if record.latency_ms is not None
        ]

        cost_complete = len(costs) == trials
        latency_complete = len(latencies) == trials

        mean_cost = sum(costs, Decimal(0)) / trials if cost_complete else None

        mean_latency = sum(latencies) / trials if latency_complete else None

        examples.append(
            MLExample(
                features=extract_features(
                    prompt=first.prompt,
                    model_id=first.model_id,
                    effort=first.effort,
                ),
                targets=MLTargets(
                    success_rate=successes / trials,
                    successes=successes,
                    trials=trials,
                    mean_cost_usd=mean_cost,
                    mean_latency_ms=mean_latency,
                    cost_observations=len(costs),
                    latency_observations=len(latencies),
                ),
                masks=TargetMasks(
                    cost=cost_complete,
                    latency=latency_complete,
                ),
                metadata=MLMetadata(
                    source=first.source,
                    task_id=first.task_id,
                    domain=first.domain,
                    family_id=first.family_id,
                    original_split=first.split,
                    group_key=task_group_key(first),
                    run_ids=sorted({record.run_id for record in group}),
                ),
            )
        )

    statistics = {
        "input_records": len(records),
        "excluded_smoke": excluded_smoke,
        "excluded_nonterminal": excluded_nonterminal,
        "terminal_records_used": sum(len(group) for group in grouped.values()),
        "aggregated_examples": len(examples),
        "complete_cost_targets": sum(example.masks.cost for example in examples),
        "complete_latency_targets": sum(example.masks.latency for example in examples),
    }

    return examples, statistics


def split_ml_dataset(
    examples: list[MLExample],
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[MLExample]]:

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    if not examples:
        raise ValueError(
            "No eligible ML examples. Collect non-smoke benchmark data first."
        )

    # ----------------------------------
    # 1. Group related tasks
    # ----------------------------------

    groups = defaultdict(list)

    for example in examples:
        groups[example.metadata.group_key].append(example)

    # ----------------------------------
    # 2. Identify predefined splits
    # ----------------------------------

    group_splits = {}

    train_groups = []
    validation_groups = []

    for group_key, group in sorted(groups.items()):
        original_splits = {example.metadata.original_split for example in group}

        # A family must not cross official splits.
        if len(original_splits) != 1:
            raise ValueError(
                f"Task family {group_key!r} crosses "
                f"predefined dataset splits: "
                f"{sorted(original_splits)}"
            )

        original_split = next(iter(original_splits))

        if original_split == "test":
            group_splits[group_key] = "test"

        elif original_split == "validation":
            validation_groups.append(group_key)

            group_splits[group_key] = "validation"

        else:
            # Includes explicitly permitted smoke data.
            train_groups.append(group_key)

    # ----------------------------------
    # 3. Create validation groups
    # ----------------------------------

    if not validation_groups:
        if len(train_groups) < 2:
            raise ValueError(
                "Need at least two distinct training "
                "task groups to create a validation split"
            )

        rng = random.Random(seed)

        shuffled = sorted(train_groups)
        rng.shuffle(shuffled)

        validation_count = round(len(shuffled) * validation_fraction)

        validation_count = max(
            1,
            min(
                validation_count,
                len(shuffled) - 1,
            ),
        )

        selected_validation = set(shuffled[:validation_count])

        for group_key in train_groups:
            group_splits[group_key] = (
                "validation" if group_key in selected_validation else "train"
            )

    else:
        # The dataset already defines validation data.
        for group_key in train_groups:
            group_splits[group_key] = "train"

    # ----------------------------------
    # 4. Build final splits
    # ----------------------------------

    splits: dict[str, list[MLExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for group_key, group in groups.items():
        destination = group_splits[group_key]

        splits[destination].extend(group)

    # ----------------------------------
    # 5. Validate split integrity
    # ----------------------------------

    if not splits["train"]:
        raise ValueError("Training split is empty")

    if not splits["validation"]:
        raise ValueError("Validation split is empty")

    split_groups = {
        split: {example.metadata.group_key for example in rows}
        for split, rows in splits.items()
    }

    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        if split_groups[left] & split_groups[right]:
            raise ValueError(f"Group leakage between {left} and {right}")

    # Deterministic output ordering
    for rows in splits.values():
        rows.sort(
            key=lambda example: (
                example.metadata.source,
                example.metadata.task_id,
                example.features.model_id,
                example.features.effort,
            )
        )

    return splits


def write_jsonl(
    path: Path,
    examples: list[MLExample],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write to a temporary file first.
    temporary_path = path.with_suffix(".jsonl.tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for example in examples:
            file.write(example.model_dump_json() + "\n")

    temporary_path.replace(path)


def build_ml_dataset(
    input_paths: list[str | Path],
    output_dir: str | Path,
    validation_fraction: float = 0.15,
    seed: int = 42,
    include_smoke: bool = False,
) -> dict:

    output_dir = Path(output_dir)

    # ----------------------------------
    # 1. Load source observations
    # ----------------------------------

    records = load_collection_records(input_paths)

    # ----------------------------------
    # 2. Aggregate repetitions
    # ----------------------------------

    examples, statistics = aggregate_executions(
        records,
        include_smoke=include_smoke,
    )

    # ----------------------------------
    # 3. Create leakage-safe splits
    # ----------------------------------

    splits = split_ml_dataset(
        examples,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    # ----------------------------------
    # 4. Create dataset provenance
    # ----------------------------------

    input_files = []

    for input_path in input_paths:
        path = Path(input_path)

        input_files.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "include_smoke": include_smoke,
        "input_files": input_files,
        "statistics": statistics,
        "splits": {
            split: {
                "examples": len(rows),
                "tasks": len(
                    {
                        (
                            row.metadata.source,
                            row.metadata.task_id,
                        )
                        for row in rows
                    }
                ),
                "groups": len({row.metadata.group_key for row in rows}),
                "complete_cost_targets": sum(row.masks.cost for row in rows),
                "complete_latency_targets": sum(row.masks.latency for row in rows),
            }
            for split, rows in splits.items()
        },
        "feature_fields": list(MLFeatures.model_fields),
        "target_fields": list(MLTargets.model_fields),
    }

    # ----------------------------------
    # 5. Write final datasets
    # ----------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split, rows in splits.items():
        write_jsonl(
            output_dir / f"{split}.jsonl",
            rows,
        )

    manifest_path = output_dir / "manifest.json"

    temporary_manifest = output_dir / "manifest.json.tmp"

    temporary_manifest.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_manifest.replace(manifest_path)

    return manifest
