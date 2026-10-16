from pathlib import Path

from pydantic import ValidationError

from backend.benchmarks.schema import (
    BenchmarkTask,
    DatasetSplit,
)


def load_tasks(
    path: str | Path,
    split: DatasetSplit | None = None,
) -> list[BenchmarkTask]:
    """
    Load and validate benchmark tasks from JSONL.

    Args:
        path: Path to the JSONL dataset.
        split: Optional dataset split filter.

    Returns:
        List of validated benchmark tasks.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Benchmark dataset not found: {path}")

    tasks = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                task = BenchmarkTask.model_validate_json(line)

            except ValidationError as exc:
                raise ValueError(
                    f"Invalid benchmark task at {path}:{line_number}: {exc}"
                ) from exc

            # Prevent duplicate task IDs
            if task.task_id in seen_ids:
                raise ValueError(
                    f"Duplicate task ID '{task.task_id}' at {path}:{line_number}"
                )

            seen_ids.add(task.task_id)

            if split is not None and task.split != split:
                continue

            tasks.append(task)

    return tasks
