import argparse
import asyncio
import fcntl
from decimal import Decimal
from pathlib import Path

from backend.benchmarks.collection import (
    CollectionRunner,
    estimate_attempt_cost,
)
from backend.benchmarks.exporter import export_collection
from backend.benchmarks.loader import load_tasks
from backend.benchmarks.runner import ModelEffort
from backend.core.model_registry import ModelRegistry
from backend.providers.openai_provider import OpenAIProvider
from backend.services.execution_service import ExecutionService


def parse_args():
    parser = argparse.ArgumentParser(description="BrainRoute Dataset Collection")

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--models",
        nargs="+",
    )

    parser.add_argument(
        "--efforts",
        nargs="+",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-calls",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--budget",
        type=Decimal,
        default=Decimal("0.10"),
    )

    parser.add_argument(
        "--resume",
        type=int,
        default=None,
        metavar="RUN_ID",
    )

    parser.add_argument(
        "--export",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Permit real, billable API requests",
    )

    parser.add_argument(
        "--increase-budget-to",
        type=Decimal,
        default=None,
    )

    return parser.parse_args()


def configurations_from_args(args):
    return [
        ModelEffort(
            model_id=model_id,
            effort=effort,
        )
        for model_id in args.models
        for effort in args.efforts
    ]


async def main():
    args = parse_args()

    registry = ModelRegistry()
    tasks = load_tasks(args.dataset)

    if args.resume is None and args.increase_budget_to is not None:
        raise ValueError("--increase-budget-to requires --resume")

    print("\n========== BRAINROUTE COLLECTOR ==========")
    print("Dataset:", args.dataset)
    print("Tasks:", len(tasks))

    configurations = None

    if args.resume is None:
        if not args.models or not args.efforts:
            raise ValueError("--models and --efforts are required for new experiments")

        configurations = configurations_from_args(args)

        if not 1 <= args.repeats <= 10:
            raise ValueError("Invalid repetition count")

        if not 16 <= args.max_output_tokens <= 4096:
            raise ValueError("Invalid output-token limit")

        planned_calls = len(tasks) * len(configurations) * args.repeats

        if planned_calls > args.max_calls:
            raise ValueError(
                f"Planned {planned_calls} calls exceed max-calls={args.max_calls}"
            )

        estimated_total = Decimal(0)

        for task in tasks:
            for config in configurations:
                model = registry.validate_configuration(
                    config.model_id,
                    config.effort,
                )

                if model.provider != "openai":
                    raise ValueError("This CLI currently supports OpenAI only")

                estimated_total += (
                    estimate_attempt_cost(
                        task,
                        config,
                        registry,
                        args.max_output_tokens,
                    )
                    * args.repeats
                )

        print("Configurations:", len(configurations))
        print("Repetitions:", args.repeats)
        print("Planned executions:", planned_calls)
        print("Budget:", args.budget, "USD")
        print(
            "Estimated full-plan cost:",
            estimated_total,
            "USD",
        )

        if estimated_total > args.budget:
            print("Warning: The collector may pause before completing this plan.")

    else:
        if args.models or args.efforts:
            raise ValueError(
                "Do not provide models/efforts with --resume; "
                "the original plan is loaded from PostgreSQL."
            )

        print("Resuming run:", args.resume)

    if not args.live:
        print("\nDRY RUN — no API requests made")
        print("Add --live to start or resume collection.")
        return

    # Local macOS process lock.
    #
    # This prevents two instances of this CLI in the
    # same project directory from collecting simultaneously.
    lock_path = Path(".brainroute_collection.lock")

    with lock_path.open("a+") as lock_file:  # noqa: ASYNC230
        try:
            fcntl.flock(
                lock_file,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another BrainRoute collector is already running"
            ) from exc

        provider = OpenAIProvider(registry=registry)

        try:
            service = ExecutionService(
                provider=provider,
                registry=registry,
            )

            collector = CollectionRunner(
                execution_service=service,
                registry=registry,
            )

            summary = await collector.collect(
                dataset_path=args.dataset,
                configurations=configurations,
                repeats=args.repeats,
                seed=args.seed,
                max_calls=args.max_calls,
                max_output_tokens=args.max_output_tokens,
                budget_usd=args.budget,
                resume_run_id=args.resume,
                on_progress=print,
                increase_budget_to=args.increase_budget_to,
            )

            print("\n========== COLLECTION SUMMARY ==========")
            print("Run ID:", summary.run_id)
            print("Status:", summary.status)
            print("Total:", summary.total)
            print("Completed:", summary.completed)
            print("Errors:", summary.errors)
            print("Uncertain:", summary.uncertain)
            print("Pending:", summary.pending)

            print(
                "Known expenditure:",
                summary.known_cost_usd,
                "USD",
            )

            print(
                "Attempts with unknown costs:",
                summary.unknown_costs,
            )

            output_path = args.export or Path(
                f"datasets/exports/run_{summary.run_id}.jsonl"
            )

            rows = export_collection(
                run_id=summary.run_id,
                dataset_path=args.dataset,
                output_path=output_path,
            )

            print("\nExported records:", rows)
            print("Export file:", output_path)

        finally:
            await provider.close()

            fcntl.flock(
                lock_file,
                fcntl.LOCK_UN,
            )


if __name__ == "__main__":
    asyncio.run(main())
