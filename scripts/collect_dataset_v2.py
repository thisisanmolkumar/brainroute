"""Phase 4.3: approval-gated Dataset v2 collector over existing persisted runner.

Dry-run is the default. No provider client is created without --live and an
exact plan-ID approval. Do not use this to evaluate MBPP/IFEval or test splits.
"""

import argparse
import asyncio
import fcntl
from pathlib import Path

from backend.benchmarks.collection import CollectionRunner, estimate_attempt_cost
from backend.benchmarks.collection_repository import CollectionRepository
from backend.benchmarks.runner import ModelEffort
from backend.core.model_registry import ModelRegistry
from backend.datasets.v2_bridge import (
    export_collection_v2,
    materialize_benchmark_dataset,
    validate_collection_plan,
)
from backend.providers.openai_provider import OpenAIProvider
from backend.services.execution_service import ExecutionService


def parse_args():
    parser = argparse.ArgumentParser(
        description="BrainRoute Phase 4.3 — Dataset v2 collection"
    )
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--resume", type=int, default=None, metavar="RUN_ID")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--approve-plan-id", default=None)
    parser.add_argument("--export", type=Path, default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    plan = validate_collection_plan(args.plan_dir, args.pool)
    registry = ModelRegistry()

    estimates = []
    for task in plan.benchmark_tasks:
        for model_id, effort in plan.configurations:
            config = ModelEffort(model_id, effort)
            model = registry.validate_configuration(model_id, effort)
            if model.provider != "openai":
                raise ValueError("Dataset v2 CLI supports OpenAI only")
            estimate = estimate_attempt_cost(
                task, config, registry, plan.max_output_tokens
            )
            if estimate > plan.attempt_cap_usd:
                raise ValueError(
                    "Estimated cost exceeds per-attempt limit for "
                    f"{task.task_id} / {model_id}:{effort}"
                )
            estimates.append(estimate * plan.repeats)

    print("\n========== DATASET V2 COLLECTION ==========")
    print("Plan ID:", plan.plan_id)
    print("Tasks:", len(plan.benchmark_tasks))
    print("Calls:", len(plan.attempts))
    print("Budget (USD):", plan.budget_usd)
    print("Maximum estimated cost per call (USD):", plan.attempt_cap_usd)
    print("Estimated plan cost (USD):", sum(estimates))
    print("Timeout (seconds):", plan.timeout_seconds)

    if not args.live:
        print("DRY RUN — no provider calls made and no collection run created")
        return
    if args.approve_plan_id != plan.plan_id:
        raise ValueError(
            "Live execution requires --approve-plan-id matching the exact plan ID"
        )

    # Identical lock and repository as Phase 2 collection.
    lock_path = Path(".brainroute_collection.lock")
    with lock_path.open("a+") as lock_file:  # noqa: ASYNC230
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another collection process is running") from exc

        repository = CollectionRepository()
        existing_run_id = repository.find_run_by_v2_plan_id(plan.plan_id)
        if args.resume is None and existing_run_id is not None:
            raise ValueError(
                "This plan already has a persisted collection run "
                f"({existing_run_id}); use --resume {existing_run_id}"
            )
        if args.resume is not None and existing_run_id != args.resume:
            raise ValueError("Resume run does not belong to this exact v2 plan")

        materialize_benchmark_dataset(plan, create=args.resume is None)
        provider = OpenAIProvider(registry=registry)
        try:
            service = ExecutionService(provider=provider, registry=registry)
            runner = CollectionRunner(
                execution_service=service, registry=registry, repository=repository
            )
            summary = await runner.collect(
                dataset_path=plan.dataset_path,
                configurations=[
                    ModelEffort(model_id, effort)
                    for model_id, effort in plan.configurations
                ]
                if args.resume is None
                else None,
                repeats=plan.repeats,
                seed=plan.seed,
                max_calls=len(plan.attempts),
                max_output_tokens=plan.max_output_tokens,
                budget_usd=plan.budget_usd,
                timeout_seconds=plan.timeout_seconds,
                max_estimated_attempt_cost_usd=plan.attempt_cap_usd,
                plan_metadata=plan.metadata,
                resume_run_id=args.resume,
                on_progress=print,
            )
            print("Run ID:", summary.run_id)
            print("Status:", summary.status)
            print("Completed:", summary.completed)
            print("Uncertain:", summary.uncertain)
            print("Known spend (USD):", summary.known_cost_usd)

            output_path = args.export or Path(
                "datasets/exports/"
                f"v2_run_{summary.run_id}_completed_{summary.completed}"
                f"_pending_{summary.pending}.jsonl"
            )
            rows = export_collection_v2(plan, summary.run_id, output_path, repository)
            print("Exported v2 observations:", rows)
            print("Export path:", output_path)
        finally:
            await provider.close()
            fcntl.flock(lock_file, fcntl.LOCK_UN)


if __name__ == "__main__":
    asyncio.run(main())
