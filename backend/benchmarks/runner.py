import asyncio
import hashlib
import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from backend.benchmarks.repository import BenchmarkRepository
from backend.benchmarks.schema import BenchmarkTask
from backend.benchmarks.verifiers import VerificationService
from backend.core.model_registry import (
    ModelRegistry,
    ReasoningEffort,
)
from backend.schemas.llm import LLMRequest
from backend.services.execution_service import ExecutionService

logger = logging.getLogger(__name__)


MILLION = Decimal(1000000)


@dataclass(frozen=True)
class ModelEffort:
    model_id: str
    effort: ReasoningEffort


@dataclass
class AttemptOutcome:
    task_id: str
    model_id: str
    effort: str

    status: str
    verification_status: str | None
    correct: bool | None

    execution_id: int | None
    cost_usd: Decimal | None

    error_type: str | None = None


@dataclass
class BenchmarkResult:
    run_id: int
    outcomes: list[AttemptOutcome]
    estimated_max_cost_usd: Decimal


class BenchmarkRunner:
    def __init__(
        self,
        execution_service: ExecutionService,
        registry: ModelRegistry,
        repository: BenchmarkRepository | None = None,
        verifier: VerificationService | None = None,
    ):
        self.execution_service = execution_service
        self.registry = registry

        self.repository = repository or BenchmarkRepository()

        self.verifier = verifier or VerificationService()

    def estimate_plan_cost(
        self,
        tasks: list[BenchmarkTask],
        configurations: list[ModelEffort],
        max_output_tokens: int,
    ) -> Decimal:
        """
        Conservative pricing estimate.

        This is NOT a hard billing guarantee.
        """

        total = Decimal(0)

        for task in tasks:
            for config in configurations:
                model = self.registry.validate_configuration(
                    config.model_id,
                    config.effort,
                )

                if model.provider != self.execution_service.provider.name:
                    raise ValueError(f"Unsupported provider for {config.model_id}")

                pricing = model.pricing

                # Rough upper-biased input-token estimate.
                estimated_input = len(task.prompt.encode("utf-8")) + 512

                context_limit = pricing.short_context_max_input_tokens

                if context_limit is not None and estimated_input > context_limit:
                    raise ValueError("Task exceeds supported pricing range")

                input_rates = [
                    pricing.input_per_1m_tokens,
                ]

                if pricing.cached_input_per_1m_tokens is not None:
                    input_rates.append(pricing.cached_input_per_1m_tokens)

                if pricing.cache_write_per_1m_tokens is not None:
                    input_rates.append(pricing.cache_write_per_1m_tokens)

                input_rate = max(input_rates)

                estimated_cost = (
                    Decimal(estimated_input) * input_rate
                    + Decimal(max_output_tokens) * pricing.output_per_1m_tokens
                ) / MILLION

                total += estimated_cost

        return total

    async def _run_one(
        self,
        run_id: int,
        task: BenchmarkTask,
        config: ModelEffort,
        max_output_tokens: int,
        semaphore: asyncio.Semaphore,
    ) -> AttemptOutcome:

        async with semaphore:
            attempt_id = await asyncio.to_thread(
                self.repository.create_attempt,
                run_id,
                task.task_id,
                config.model_id,
                config.effort,
            )

            execution_id = None

            async def on_created(new_execution_id: int):
                nonlocal execution_id

                await asyncio.to_thread(
                    self.repository.attach_execution,
                    attempt_id,
                    new_execution_id,
                )

                execution_id = new_execution_id

            request = LLMRequest(
                prompt=task.prompt,
                model_id=config.model_id,
                effort=config.effort,
                max_output_tokens=max_output_tokens,
            )

            try:
                result = await self.execution_service.execute(
                    request=request,
                    category=task.domain,
                    on_created=on_created,
                )

                verification = self.verifier.verify(
                    task=task,
                    output_text=result.response.output_text,
                    execution_status=result.response.status,
                )

                await asyncio.to_thread(
                    self.repository.complete_attempt,
                    attempt_id,
                    verification,
                )

                cost = None

                if result.metrics.cost is not None:
                    cost = result.metrics.cost.total_cost_usd

                return AttemptOutcome(
                    task_id=task.task_id,
                    model_id=config.model_id,
                    effort=config.effort,
                    status="completed",
                    verification_status=verification.status,
                    correct=verification.correct,
                    execution_id=result.execution_id,
                    cost_usd=cost,
                )

            except Exception as exc:
                logger.exception(
                    "Benchmark execution failed: task=%s, model=%s, effort=%s",
                    task.task_id,
                    config.model_id,
                    config.effort,
                )

                await asyncio.to_thread(
                    self.repository.fail_attempt,
                    attempt_id,
                    exc,
                )

                return AttemptOutcome(
                    task_id=task.task_id,
                    model_id=config.model_id,
                    effort=config.effort,
                    status="error",
                    verification_status=None,
                    correct=None,
                    execution_id=execution_id,
                    cost_usd=None,
                    error_type=type(exc).__name__,
                )

    async def run(
        self,
        tasks: list[BenchmarkTask],
        configurations: list[ModelEffort],
        dataset_path: str | Path,
        max_calls: int = 6,
        max_output_tokens: int = 1024,
        concurrency: int = 1,
        max_estimated_cost_usd: Decimal = Decimal("0.10"),
    ) -> BenchmarkResult:

        if not tasks:
            raise ValueError("No benchmark tasks supplied")

        if not configurations:
            raise ValueError("No configurations supplied")

        if concurrency < 1 or concurrency > 4:
            raise ValueError("Concurrency must be between 1 and 4")

        if max_calls < 1:
            raise ValueError("max_calls must be positive")

        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

        if max_estimated_cost_usd <= 0:
            raise ValueError("Estimated cost budget must be positive")

        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("Duplicate benchmark task IDs")

        if len(set(configurations)) != len(configurations):
            raise ValueError("Duplicate model configurations")

        total_calls = len(tasks) * len(configurations)

        if total_calls > max_calls:
            raise ValueError(
                f"Planned {total_calls} requests exceeds max_calls={max_calls}"
            )

        # Validate all configurations and estimate
        # expenditure before creating any records.

        estimated_cost = self.estimate_plan_cost(
            tasks,
            configurations,
            max_output_tokens,
        )

        if estimated_cost > max_estimated_cost_usd:
            raise ValueError(
                f"Estimated cost ${estimated_cost} exceeds "
                f"budget ${max_estimated_cost_usd}"
            )

        path = Path(dataset_path)

        if not path.is_file():
            raise FileNotFoundError(path)

        dataset_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

        # Create benchmark run.
        run_id = await asyncio.to_thread(
            self.repository.create_run,
            str(path.resolve()),
            dataset_sha256,
        )

        semaphore = asyncio.Semaphore(concurrency)

        try:
            async with asyncio.TaskGroup() as group:
                jobs = [
                    group.create_task(
                        self._run_one(
                            run_id,
                            task,
                            config,
                            max_output_tokens,
                            semaphore,
                        )
                    )
                    for task in tasks
                    for config in configurations
                ]

            outcomes = [job.result() for job in jobs]

            run_status = (
                "completed_with_errors"
                if any(outcome.status == "error" for outcome in outcomes)
                else "completed"
            )

            await asyncio.to_thread(
                self.repository.finish_run,
                run_id,
                run_status,
            )

        except BaseException:
            await asyncio.to_thread(
                self.repository.finish_run,
                run_id,
                "interrupted",
            )

            raise

        return BenchmarkResult(
            run_id=run_id,
            outcomes=outcomes,
            estimated_max_cost_usd=estimated_cost,
        )
