from pathlib import Path

from pydantic import ValidationError

from backend.benchmarks.ml_dataset import MLExample


def load_ml_examples(
    path: str | Path,
) -> list[MLExample]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    examples = []
    seen = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                example = MLExample.model_validate_json(line)

            except ValidationError as exc:
                raise ValueError(f"Invalid ML example at {path}:{line_number}") from exc

            key = (
                example.metadata.source,
                example.metadata.task_id,
                example.features.model_id,
                example.features.effort,
            )

            if key in seen:
                raise ValueError(f"Duplicate ML example: {key}")

            seen.add(key)

            targets = example.targets

            expected_rate = targets.successes / targets.trials

            if targets.successes > targets.trials:
                raise ValueError(f"Successes exceed trials for {key}")

            if abs(targets.success_rate - expected_rate) > 1e-6:
                raise ValueError(f"Inconsistent success rate for {key}")

            if example.masks.cost != (
                targets.mean_cost_usd is not None
                and targets.cost_observations == targets.trials
            ):
                raise ValueError(f"Inconsistent cost mask for {key}")

            if example.masks.latency != (
                targets.mean_latency_ms is not None
                and targets.latency_observations == targets.trials
            ):
                raise ValueError(f"Inconsistent latency mask for {key}")

            examples.append(example)

    if not examples:
        raise ValueError(f"Empty ML dataset: {path}")

    return examples


def validate_training_splits(
    train: list[MLExample],
    validation: list[MLExample],
) -> None:
    train_groups = {example.metadata.group_key for example in train}

    validation_groups = {example.metadata.group_key for example in validation}

    overlap = train_groups & validation_groups

    if overlap:
        raise ValueError(
            f"Task-group leakage between train and validation: {sorted(overlap)[:5]}"
        )

    # Preserve original official-test isolation.
    for example in [*train, *validation]:
        if example.metadata.original_split == "test":
            raise ValueError(
                "Official test examples cannot be used during baseline training"
            )

    total_trials = sum(example.targets.trials for example in train)

    total_successes = sum(example.targets.successes for example in train)

    if not 0 < total_successes < total_trials:
        raise ValueError(
            "Training data contains only one outcome class. "
            "Collect both successful and unsuccessful attempts."
        )


def group_count(examples: list[MLExample]) -> int:
    return len({example.metadata.group_key for example in examples})
