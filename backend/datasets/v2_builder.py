import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from backend.datasets.v2_schema import TaskV2
from backend.ml.data import load_ml_examples

SPLITS = (
    "train",
    "validation",
    "calibration",
    "test",
)


def normalize_prompt(prompt: str) -> str:
    text = unicodedata.normalize("NFKC", prompt)
    return " ".join(text.casefold().split())


def content_hash(prompt: str) -> str:
    normalized = normalize_prompt(prompt)

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tasks(paths: list[Path]) -> list[TaskV2]:
    tasks = []
    identities = set()

    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue

                try:
                    task = TaskV2.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc

                identity = (task.source, task.task_id)

                if identity in identities:
                    raise ValueError(f"Duplicate task ID: {identity}")

                identities.add(identity)
                tasks.append(task)

    return sorted(
        tasks,
        key=lambda task: (
            task.source,
            task.task_id,
        ),
    )


def existing_split_assignments(
    paths: list[Path],
) -> dict[tuple[str, str], str]:

    assignments = {}

    for path in paths:
        examples = load_ml_examples(path)

        for item in examples:
            key = (
                item.metadata.source,
                item.metadata.task_id,
            )

            split = item.metadata.original_split

            if split == "val":
                split = "validation"

            if split not in SPLITS:
                continue

            previous = assignments.get(key)

            if previous is not None and previous != split:
                raise ValueError(f"Conflicting existing splits: {key}")

            assignments[key] = split

    return assignments


def build_groups(
    tasks: list[TaskV2],
) -> list[list[TaskV2]]:
    """
    Connect tasks sharing either a family or
    normalized prompt. Connected tasks must
    remain in the same dataset split.
    """

    parent = list(range(len(tasks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]

        return index

    def union(first: int, second: int) -> None:
        root_a = find(first)
        root_b = find(second)

        if root_a != root_b:
            parent[root_b] = root_a

    family_owner = {}
    prompt_owner = {}

    for index, task in enumerate(tasks):
        family = (
            task.source,
            task.family_id,
        )

        fingerprint = content_hash(task.raw_prompt)

        if family in family_owner:
            union(index, family_owner[family])
        else:
            family_owner[family] = index

        if fingerprint in prompt_owner:
            union(index, prompt_owner[fingerprint])
        else:
            prompt_owner[fingerprint] = index

    grouped = defaultdict(list)

    for index, task in enumerate(tasks):
        grouped[find(index)].append(task)

    return list(grouped.values())


def assigned_split(
    tasks: list[TaskV2],
    locked: dict[tuple[str, str], str],
    seed: int,
) -> str:

    source_splits = {task.source_split for task in tasks}

    # If official source partitions overlap,
    # reject the connected family rather than
    # silently leaking information.
    if len(source_splits) != 1:
        raise ValueError(
            f"Related tasks cross official source splits: {sorted(source_splits)}"
        )

    official_split = next(iter(source_splits))

    existing = {
        locked[(task.source, task.task_id)]
        for task in tasks
        if (task.source, task.task_id) in locked
    }

    if len(existing) > 1:
        raise ValueError("Related tasks have conflicting historical split assignments")

    locked_split = next(iter(existing), None)

    if official_split == "test":
        proposed = "test"

    elif official_split == "validation":
        proposed = "validation"

    else:
        family_keys = sorted({f"{task.source}:{task.family_id}" for task in tasks})

        group_identity = "|".join(family_keys)

        digest = hashlib.sha256(f"{seed}:{group_identity}".encode()).digest()

        fraction = int.from_bytes(digest[:8], "big") / (1 << 64)

        if fraction < 0.70:
            proposed = "train"
        elif fraction < 0.80:
            proposed = "validation"
        elif fraction < 0.90:
            proposed = "calibration"
        else:
            proposed = "test"

    if locked_split is not None:
        if official_split == "test" and locked_split != "test":
            raise ValueError(
                "An official test task was previously used outside the test split"
            )

        if official_split == "validation" and locked_split != "validation":
            raise ValueError(
                "Official validation split conflicts with historical assignment"
            )

        return locked_split

    return proposed


def compile_dataset(
    input_paths: list[Path],
    output_dir: Path,
    *,
    seed: int = 42,
    existing_ml_paths: list[Path] | None = None,
) -> dict:

    if not input_paths:
        raise ValueError("No source files provided")

    tasks = load_tasks(input_paths)

    if not tasks:
        raise ValueError("No valid source tasks found")

    locked = existing_split_assignments(existing_ml_paths or [])

    groups = build_groups(tasks)

    split_tasks: dict[str, list[TaskV2]] = {name: [] for name in SPLITS}

    split_group_counts = Counter()
    duplicates_removed = 0

    for group in groups:
        split = assigned_split(group, locked, seed)
        split_group_counts[split] += 1

        # Detect identical prompts with contradictory
        # labels before removing duplicates.
        unique = {}

        for task in group:
            fingerprint = content_hash(task.raw_prompt)

            previous = unique.get(fingerprint)

            if previous is not None:
                if (
                    previous.reference_answer is not None
                    and task.reference_answer is not None
                    and previous.reference_answer != task.reference_answer
                ):
                    raise ValueError(
                        "Conflicting reference answers for "
                        f"identical prompts: {task.task_id}"
                    )

                if (
                    previous.verifier is not None
                    and task.verifier is not None
                    and previous.verifier != task.verifier
                ):
                    raise ValueError(
                        f"Conflicting verifiers for identical prompts: {task.task_id}"
                    )

                duplicates_removed += 1
                continue

            unique[fingerprint] = task

        split_tasks[split].extend(unique.values())

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_reports = {}

    for split, rows in split_tasks.items():
        rows.sort(
            key=lambda task: (
                task.source,
                task.task_id,
            )
        )

        output_path = output_dir / f"pool_{split}.jsonl"

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as stream:
            for task in rows:
                payload = task.model_dump(mode="json")

                stream.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        split_reports[split] = {
            "tasks": len(rows),
            "groups": split_group_counts[split],
            "ready": sum(item.verification_status == "ready" for item in rows),
            "pending": sum(item.verification_status == "pending" for item in rows),
            "domains": dict(sorted(Counter(item.domain for item in rows).items())),
            "difficulties": dict(
                sorted(Counter(item.difficulty for item in rows).items())
            ),
            "sources": dict(sorted(Counter(item.source for item in rows).items())),
            "sha256": file_hash(output_path),
        }

    manifest = {
        "schema_version": 2,
        "dataset_name": "brainroute_dataset_v2",
        "seed": seed,
        "source_files": [
            {
                "path": str(path),
                "sha256": file_hash(path),
            }
            for path in input_paths
        ],
        "source_tasks": len(tasks),
        "connected_task_groups": len(groups),
        "exact_duplicates_removed": duplicates_removed,
        "compiled_tasks": sum(item["tasks"] for item in split_reports.values()),
        "splits": split_reports,
        "warnings": [
            "Pending tasks are not authorized for benchmark collection.",
            (
                "Near-duplicate or semantically equivalent "
                "tasks may require manual review."
            ),
            "Hash splitting does not guarantee domain-balanced partitions.",
            (
                "The original official test partition "
                "must remain untouched during training."
            ),
        ],
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest
