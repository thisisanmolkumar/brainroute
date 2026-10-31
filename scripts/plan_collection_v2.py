import argparse
import hashlib
import json
from pathlib import Path

from backend.datasets.v2_builder import load_tasks
from backend.datasets.v2_collection import (
    build_collection_plan,
)
from backend.ml.data import load_ml_examples


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan BrainRoute Dataset v2 collection"
    )

    parser.add_argument(
        "--pool",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--legacy-data",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--budget-usd",
        required=True,
    )
    parser.add_argument(
        "--per-attempt-cap-usd",
        required=True,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.pool.name != "pool_train.jsonl":
        raise ValueError("Phase 4.3 training collection requires pool_train.jsonl")

    # Verify the compiled Dataset v2 pool before creating an immutable plan.
    compiled_manifest_path = args.pool.parent / "manifest.json"
    compiled_manifest = json.loads(compiled_manifest_path.read_text(encoding="utf-8"))
    if args.pool.name != "pool_train.jsonl":
        raise ValueError("Only the compiled training pool can be collected")
    if sha256_file(args.pool) != compiled_manifest["splits"]["train"]["sha256"]:
        raise ValueError("Dataset pool does not match its manifest")

    tasks = load_tasks([args.pool])

    legacy_prompts = []

    for path in args.legacy_data:
        examples = load_ml_examples(path)

        legacy_prompts.extend(example.features.prompt for example in examples)

    plan = build_collection_plan(
        tasks,
        candidates=args.candidate,
        legacy_prompts=legacy_prompts,
        max_tasks=args.max_tasks,
        repeats=args.repeats,
        budget_usd=args.budget_usd,
        per_attempt_cap_usd=(args.per_attempt_cap_usd),
        max_output_tokens=(args.max_output_tokens),
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
    )

    # Refuse to overwrite an existing experiment.
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    attempts_path = args.output_dir / "attempts.jsonl"

    with attempts_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for attempt in plan["attempts"]:
            stream.write(
                json.dumps(
                    attempt,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {key: value for key, value in plan.items() if key != "attempts"}

    manifest["pool_sha256"] = sha256_file(args.pool)
    manifest["legacy_data_sha256"] = {
        str(path): sha256_file(path) for path in args.legacy_data
    }
    manifest["attempts_sha256"] = sha256_file(attempts_path)

    manifest_path = args.output_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nBrainRoute Phase 4.3 Plan")
    print("Plan ID:", plan["plan_id"])
    print("Selected tasks:", plan["selected_tasks"])
    print("Planned calls:", plan["planned_attempts"])
    print("Budget:", plan["budget_usd"], "USD")
    print("Excluded:", plan["excluded"])
    print("Output:", args.output_dir)
    print("Paid execution: DISABLED")


if __name__ == "__main__":
    main()
