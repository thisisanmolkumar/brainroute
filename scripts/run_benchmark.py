import argparse
import asyncio
from decimal import Decimal
from pathlib import Path

from backend.benchmarks.loader import load_tasks
from backend.benchmarks.runner import (
    BenchmarkRunner,
    ModelEffort,
)
from backend.core.model_registry import ModelRegistry
from backend.providers.openai_provider import OpenAIProvider
from backend.services.execution_service import ExecutionService


def parse_args():
    parser = argparse.ArgumentParser(description="BrainRoute Benchmark Runner")

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--efforts",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--max-calls",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-estimated-cost-usd",
        type=Decimal,
        default=Decimal("0.10"),
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually execute paid API requests",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    registry = ModelRegistry()
    tasks = load_tasks(args.dataset)

    configurations = [
        ModelEffort(
            model_id=model_id,
            effort=effort,
        )
        for model_id in args.models
        for effort in args.efforts
    ]

    total_calls = len(tasks) * len(configurations)

    print("\n========== BRAINROUTE BENCHMARK ==========")
    print("Dataset:", args.dataset)
    print("Tasks:", len(tasks))
    print("Configurations:", len(configurations))
    print("Planned executions:", total_calls)
    print("Concurrency:", args.concurrency)

    # Dry-run mode does not require an API key.
    if not args.live:
        print("\nDRY RUN — no API calls will be made")

        for config in configurations:
            model = registry.validate_configuration(
                config.model_id,
                config.effort,
            )

            if model.provider != "openai":
                raise ValueError("Only OpenAI is supported by this CLI")

            print(f"  {config.model_id} / {config.effort}")

        if total_calls > args.max_calls:
            raise ValueError("Planned executions exceed --max-calls")

        print("\nAdd --live to execute this benchmark.")
        return

    provider = OpenAIProvider(registry=registry)

    try:
        service = ExecutionService(
            provider=provider,
            registry=registry,
        )

        runner = BenchmarkRunner(
            execution_service=service,
            registry=registry,
        )

        result = await runner.run(
            tasks=tasks,
            configurations=configurations,
            dataset_path=args.dataset,
            max_calls=args.max_calls,
            max_output_tokens=args.max_output_tokens,
            concurrency=args.concurrency,
            max_estimated_cost_usd=(args.max_estimated_cost_usd),
        )

        print("\n========== RESULTS ==========")
        print("Run ID:", result.run_id)

        correct = 0
        known_cost = Decimal(0)
        unknown_costs = 0

        for outcome in result.outcomes:
            if outcome.correct is True:
                correct += 1

            if outcome.cost_usd is not None:
                known_cost += outcome.cost_usd
            else:
                unknown_costs += 1

            label = outcome.verification_status or outcome.error_type or outcome.status

            print(
                f"{outcome.task_id} | {outcome.model_id} | {outcome.effort} | {label}"
            )

        total = len(result.outcomes)

        print("\n========== SUMMARY ==========")
        print("Total evaluations:", total)
        print("Correct:", correct)

        if total:
            print(f"Strict success rate: {correct / total:.2%}")

        print(f"Known execution cost: ${known_cost:.8f}")
        print("Unknown costs:", unknown_costs)

        print(
            "Estimated plan cost:",
            result.estimated_max_cost_usd,
            "USD",
        )

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
