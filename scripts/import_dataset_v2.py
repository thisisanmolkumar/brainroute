import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from backend.datasets.v2_adapters import adapt_record
from backend.datasets.v2_readiness import assert_collectible

DATASETS = {
    "gsm8k": {
        "repo": "openai/gsm8k",
        "config": "main",
        "split": "train",
    },
    "mbpp": {
        "repo": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": "train",
    },
    "bbh": {
        "repo": "lukaemon/bbh",
        "config": "boolean_expressions",
        "split": "test",
    },
    "ifeval": {
        "repo": "google/IFEval",
        "config": None,
        "split": "train",
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_records(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue

            item = json.loads(line)

            if not isinstance(item, dict):
                raise TypeError(f"{path}:{line_number}: expected object")

            records.append(item)

    return records


def remote_records(
    repo: str,
    config: str | None,
    split: str,
    revision: str | None,
):
    from huggingface_hub import HfApi

    from datasets import load_dataset

    info = HfApi().dataset_info(
        repo_id=repo,
        revision=revision,
    )

    pinned_revision = info.sha

    if not pinned_revision:
        raise ValueError("Unable to resolve dataset commit SHA")

    print(
        f"[4.2] Loading {repo}@{pinned_revision}",
        flush=True,
    )

    dataset = load_dataset(
        repo,
        name=config,
        split=split,
        revision=pinned_revision,
    )

    return dataset, pinned_revision


def main():
    parser = argparse.ArgumentParser(
        description=("Import multi-domain benchmarks into BrainRoute Dataset v2")
    )

    parser.add_argument(
        "--source",
        choices=sorted(DATASETS),
        required=True,
    )

    parser.add_argument("--config")
    parser.add_argument("--split")
    parser.add_argument("--revision")

    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help="Optional local source for offline testing",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum records; 0 means all",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    settings = DATASETS[args.source]

    config = args.config if args.config is not None else settings["config"]

    split = args.split if args.split is not None else settings["split"]

    if args.input_jsonl is not None:
        if not args.revision:
            raise ValueError("Local imports require --revision")

        records = local_records(args.input_jsonl)
        resolved_revision = args.revision

        source_fingerprint = sha256_file(args.input_jsonl)

    else:
        records, resolved_revision = remote_records(
            settings["repo"],
            config,
            split,
            args.revision,
        )

        source_fingerprint = None

    tasks = []

    for index, record in enumerate(records):
        if args.limit and index >= args.limit:
            break

        task = adapt_record(
            args.source,
            record,
            revision=resolved_revision,
            split=split,
            index=index,
            config=config,
        )

        tasks.append(task)

    if not tasks:
        raise ValueError("No tasks imported")

    identities = [(task.source, task.task_id) for task in tasks]

    if len(identities) != len(set(identities)):
        raise ValueError("Source produced duplicate task IDs")

    # All tasks marked ready must have a
    # registered, permitted verification strategy.
    for task in tasks:
        if task.verification_status == "ready":
            assert_collectible(task)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for task in tasks:
            stream.write(task.model_dump_json() + "\n")

    status_counts = Counter(task.verification_status for task in tasks)

    split_counts = Counter(task.source_split for task in tasks)

    manifest = {
        "schema_version": 2,
        "source": args.source,
        "upstream_dataset": settings["repo"],
        "upstream_config": config,
        "upstream_split": split,
        "source_revision": resolved_revision,
        "local_source_sha256": source_fingerprint,
        "imported_tasks": len(tasks),
        "ready_tasks": status_counts["ready"],
        "pending_tasks": status_counts["pending"],
        "source_splits": dict(split_counts),
        "output_sha256": sha256_file(args.output),
        "collection_enabled": False,
    }

    manifest_path = args.output.with_suffix(".manifest.json")

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n========== DATASET V2 IMPORT ==========")
    print("Source:", args.source)
    print("Revision:", resolved_revision)
    print("Imported:", len(tasks))
    print("Verifier ready:", status_counts["ready"])
    print("Pending:", status_counts["pending"])
    print("Output:", args.output)
    print("Manifest:", manifest_path)
    print("Collection enabled: NO")


if __name__ == "__main__":
    main()
