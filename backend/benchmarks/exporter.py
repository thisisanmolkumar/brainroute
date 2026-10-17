import hashlib
import json
from pathlib import Path

from backend.benchmarks.collection_repository import (
    CollectionRepository,
)
from backend.benchmarks.loader import load_tasks


def export_collection(
    run_id: int,
    dataset_path: str | Path,
    output_path: str | Path,
    repository: CollectionRepository | None = None,
) -> int:

    repository = repository or CollectionRepository()

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    run = repository.get_run(run_id)

    if run is None:
        raise ValueError(f"Collection run {run_id} not found")

    actual_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    if actual_sha256 != run.dataset_sha256:
        raise ValueError("Dataset checksum mismatch during export")

    tasks = {task.task_id: task for task in load_tasks(dataset_path)}

    attempts = repository.list_attempts(run_id)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for attempt in attempts:
            task = tasks[attempt.task_id]

            terminal = attempt.status in {
                "completed",
                "error",
            }

            row = {
                "run_id": run_id,
                "dataset_sha256": run.dataset_sha256,
                "task_id": task.task_id,
                "domain": task.domain,
                "source": task.source,
                "split": task.split,
                "family_id": task.family_id,
                "prompt": task.prompt,
                "reference_answer": task.reference_answer,
                "verifier": task.verifier,
                "model_id": attempt.model_id,
                "effort": attempt.effort,
                "repeat_index": attempt.repeat_index,
                "attempt_status": attempt.status,
                "provider_status": attempt.provider_status,
                "verification_status": (attempt.verification_status),
                "verified_correct": attempt.correct,
                # Strict task completion:
                # Incorrect, unparseable, and failed
                # attempts are unsuccessful.
                #
                # Pending/uncertain attempts aren't
                # assigned training labels.
                "strict_success": (
                    attempt.verification_status == "correct" if terminal else None
                ),
                "execution_id": attempt.execution_id,
                "input_tokens": attempt.input_tokens,
                "output_tokens": attempt.output_tokens,
                "reasoning_tokens": attempt.reasoning_tokens,
                "cost_usd": (
                    str(attempt.cost_usd) if attempt.cost_usd is not None else None
                ),
                "latency_ms": attempt.latency_ms,
                "error_type": attempt.error_type,
            }

            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(attempts)
