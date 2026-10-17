import asyncio
import hashlib
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from backend.benchmarks.collection_repository import (
    CollectionRepository,
    SavedAttempt,
)
from backend.benchmarks.loader import load_tasks
from backend.benchmarks.runner import ModelEffort
from backend.benchmarks.schema import BenchmarkTask
from backend.benchmarks.verifiers import VerificationService
from backend.core.model_registry import ModelRegistry
from backend.schemas.llm import LLMRequest
from backend.services.execution_service import ExecutionService

logger = logging.getLogger(__name__)
MILLION = Decimal(1000000)


@dataclass(frozen=True)
class CollectionSummary:
    run_id: int
    status: str
    total: int
    completed: int
    errors: int
    uncertain: int
    pending: int
    known_cost_usd: Decimal
    unknown_costs: int


def build_schedule(
    tasks: list[BenchmarkTask],
    configurations: list[ModelEffort],
    repeats: int,
    seed: int,
) -> list[tuple[str, str, str, int]]:

    schedule = [
        (
            task.task_id,
            config.model_id,
            config.effort,
            repeat_index,
        )
        for task in tasks
        for config in configurations
        for repeat_index in range(repeats)
    ]

    rng = random.Random(seed)
    rng.shuffle(schedule)

    return schedule


def estimate_attempt_cost(
    task: BenchmarkTask,
    config: ModelEffort,
    registry: ModelRegistry,
    max_output_tokens: int,
) -> Decimal:
    model = registry.validate_configuration(
        config.model_id,
        config.effort,
    )

    pricing = model.pricing

    # Preliminary input-token estimate.
    # This is not an exact tokenizer count.
    estimated_input_tokens = len(task.prompt.encode("utf-8")) + 512

    if (
        pricing.short_context_max_input_tokens is not None
        and estimated_input_tokens > pricing.short_context_max_input_tokens
    ):
        raise ValueError("Task exceeds supported pricing range")

    input_rates = [
        pricing.input_per_1m_tokens,
    ]

    if pricing.cached_input_per_1m_tokens is not None:
        input_rates.append(pricing.cached_input_per_1m_tokens)

    if pricing.cache_write_per_1m_tokens is not None:
        input_rates.append(pricing.cache_write_per_1m_tokens)

    return (
        Decimal(estimated_input_tokens) * max(input_rates)
        + Decimal(max_output_tokens) * pricing.output_per_1m_tokens
    ) / MILLION


class CollectionRunner:
    def __init__(
        self,
        execution_service: ExecutionService,
        registry: ModelRegistry,
        repository: CollectionRepository | None = None,
        verifier: VerificationService | None = None,
    ):
        self.execution_service = execution_service
        self.registry = registry

        self.repository = repository or CollectionRepository()

        self.verifier = verifier or VerificationService()

    async def _recover(
        self,
        run_id: int,
        tasks_by_id: dict[str, BenchmarkTask],
    ) -> None:
        """
        Recover previously running attempts.

        Never blindly resend an uncertain API request.
        """

        attempts = await asyncio.to_thread(
            self.repository.list_attempts,
            run_id,
        )

        for attempt in attempts:
            if attempt.status != "running":
                continue

            if attempt.execution_id is not None and attempt.provider_status in {
                "completed",
                "incomplete",
                "failed",
                "cancelled",
            }:
                # ExecutionService already saved the
                # provider response before interruption.

                task = tasks_by_id[attempt.task_id]

                verification = self.verifier.verify(
                    task=task,
                    output_text=attempt.output_text or "",
                    execution_status=attempt.provider_status,
                )

                await asyncio.to_thread(
                    self.repository.complete_attempt,
                    attempt.id,
                    verification,
                )

                logger.info(
                    "Recovered completed execution %s",
                    attempt.execution_id,
                )

            elif (
                attempt.execution_id is not None and attempt.provider_status == "error"
            ):
                await asyncio.to_thread(
                    self.repository.fail_attempt,
                    attempt.id,
                    attempt.error_type or "ProviderError",
                )

            else:
                # A request may have reached the provider,
                # but its outcome is not confirmed.
                await asyncio.to_thread(
                    self.repository.mark_uncertain,
                    attempt.id,
                )

    @staticmethod
    def _has_unknown_costs(
        attempts: list[SavedAttempt],
    ) -> bool:
        for attempt in attempts:
            if attempt.status in {"uncertain", "running"}:
                return True

            if attempt.status in {"completed", "error"} and attempt.cost_usd is None:
                return True

        return False

    @staticmethod
    def _known_spend(
        attempts: list[SavedAttempt],
    ) -> Decimal:
        return sum(
            (attempt.cost_usd for attempt in attempts if attempt.cost_usd is not None),
            Decimal(0),
        )

    @classmethod
    def _summary(
        cls,
        run_id: int,
        status: str,
        attempts: list[SavedAttempt],
    ) -> CollectionSummary:
        return CollectionSummary(
            run_id=run_id,
            status=status,
            total=len(attempts),
            completed=sum(a.status == "completed" for a in attempts),
            errors=sum(a.status == "error" for a in attempts),
            uncertain=sum(a.status == "uncertain" for a in attempts),
            pending=sum(a.status == "pending" for a in attempts),
            known_cost_usd=cls._known_spend(attempts),
            unknown_costs=sum(
                a.status in {"completed", "error", "uncertain"} and a.cost_usd is None
                for a in attempts
            ),
        )

    async def collect(
        self,
        dataset_path: str | Path,
        configurations: list[ModelEffort] | None = None,
        repeats: int = 1,
        seed: int = 42,
        max_calls: int = 12,
        max_output_tokens: int = 1024,
        budget_usd: Decimal = Decimal("0.10"),
        resume_run_id: int | None = None,
        on_progress: Callable[[str], None] | None = None,
        increase_budget_to: Decimal | None = None,
    ) -> CollectionSummary:

        path = Path(dataset_path)
        tasks = load_tasks(path)

        tasks_by_id = {task.task_id: task for task in tasks}

        dataset_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

        if resume_run_id is None and increase_budget_to is not None:
            raise ValueError("Budget increases require an existing run")

        # ----------------------------------
        # 1. Create or resume experiment
        # ----------------------------------

        if resume_run_id is None:
            if not tasks:
                raise ValueError("Dataset is empty")

            if not configurations:
                raise ValueError("No model configurations")

            if len(set(configurations)) != len(configurations):
                raise ValueError("Duplicate configurations")

            if not 1 <= repeats <= 10:
                raise ValueError("repeats must be between 1 and 10")

            if not 16 <= max_output_tokens <= 4096:
                raise ValueError("Invalid output-token limit")

            if budget_usd <= 0:
                raise ValueError("Budget must be positive")

            schedule = build_schedule(
                tasks,
                configurations,
                repeats,
                seed,
            )

            if max_calls < 1 or len(schedule) > max_calls:
                raise ValueError(
                    f"Planned {len(schedule)} calls exceed max_calls={max_calls}"
                )

            # Validate everything before creating records.
            for task in tasks:
                for config in configurations:
                    model = self.registry.validate_configuration(
                        config.model_id,
                        config.effort,
                    )

                    if model.provider != self.execution_service.provider.name:
                        raise ValueError("Configuration uses another provider")

                    estimate_attempt_cost(
                        task,
                        config,
                        self.registry,
                        max_output_tokens,
                    )

            plan = {
                "version": 1,
                "seed": seed,
                "repeats": repeats,
                "max_output_tokens": max_output_tokens,
                "task_ids": [task.task_id for task in tasks],
                "configurations": [
                    {
                        "model_id": config.model_id,
                        "effort": config.effort,
                    }
                    for config in configurations
                ],
            }

            run_id = await asyncio.to_thread(
                self.repository.create_run,
                str(path.resolve()),
                dataset_sha256,
                plan,
                budget_usd,
                schedule,
            )

            effective_budget = budget_usd

        else:
            saved_run = await asyncio.to_thread(
                self.repository.get_run,
                resume_run_id,
            )

            if saved_run is None:
                raise ValueError("Collection run not found")

            if saved_run.dataset_sha256 != dataset_sha256:
                raise ValueError(
                    "Dataset checksum mismatch. "
                    "Refusing to resume changed benchmark data."
                )

            if saved_run.plan is None or saved_run.budget_usd is None:
                raise ValueError("Legacy benchmark runs cannot be resumed")

            plan = saved_run.plan

            if plan.get("version") != 1:
                raise ValueError("Unsupported experiment plan")

            if plan["task_ids"] != [task.task_id for task in tasks]:
                raise ValueError("Dataset task IDs differ from saved plan")

            max_output_tokens = plan["max_output_tokens"]
            effective_budget = saved_run.budget_usd
            run_id = saved_run.id

            if increase_budget_to is not None:
                if increase_budget_to <= effective_budget:
                    raise ValueError("Budget increase must exceed the current budget")

                await asyncio.to_thread(
                    self.repository.increase_budget,
                    run_id,
                    increase_budget_to,
                )

                effective_budget = increase_budget_to

            if saved_run.status in {
                "completed",
                "completed_with_errors",
            }:
                attempts = await asyncio.to_thread(
                    self.repository.list_attempts,
                    run_id,
                )

                return self._summary(
                    run_id,
                    saved_run.status,
                    attempts,
                )

            # Recover existing responses where possible.
            await self._recover(run_id, tasks_by_id)

        # ----------------------------------
        # 2. Execute only pending attempts
        # ----------------------------------

        await asyncio.to_thread(
            self.repository.update_run_status,
            run_id,
            "running",
        )

        try:
            initial_attempts = await asyncio.to_thread(
                self.repository.list_attempts,
                run_id,
            )

            for planned in initial_attempts:
                if planned.status != "pending":
                    continue

                # Refresh costs and statuses from PostgreSQL.
                attempts = await asyncio.to_thread(
                    self.repository.list_attempts,
                    run_id,
                )

                # Unknown charges require investigation.
                if self._has_unknown_costs(attempts):
                    break

                spent = self._known_spend(attempts)

                task = tasks_by_id[planned.task_id]

                config = ModelEffort(
                    model_id=planned.model_id,
                    effort=planned.effort,
                )

                estimate = estimate_attempt_cost(
                    task,
                    config,
                    self.registry,
                    max_output_tokens,
                )

                # Adaptive per-request budget check.
                if spent + estimate > effective_budget:
                    break

                claimed = await asyncio.to_thread(
                    self.repository.claim_attempt,
                    planned.id,
                )

                if not claimed:
                    continue

                async def on_created(
                    execution_id: int,
                    attempt_id: int = planned.id,
                ) -> None:
                    await asyncio.to_thread(
                        self.repository.attach_execution,
                        attempt_id,
                        execution_id,
                    )

                request = LLMRequest(
                    prompt=task.prompt,
                    model_id=config.model_id,
                    effort=config.effort,
                    max_output_tokens=max_output_tokens,
                )

                # ----------------------------------
                # 3. Execute and checkpoint
                # ----------------------------------

                try:
                    result = await self.execution_service.execute(
                        request=request,
                        category=task.domain,
                        on_created=on_created,
                    )

                except Exception as exc:
                    # Provider failures are observations.
                    # Preserve the failure, then evaluate
                    # whether another paid call is safe.
                    logger.exception(
                        "Collection attempt %s failed",
                        planned.id,
                    )

                    await asyncio.to_thread(
                        self.repository.fail_attempt,
                        planned.id,
                        type(exc).__name__,
                    )

                    label = type(exc).__name__

                else:
                    # Verification and DB updates are
                    # deliberately outside the catch block.
                    # If they fail, recovery can reuse the
                    # existing provider response.
                    verification = self.verifier.verify(
                        task=task,
                        output_text=result.response.output_text,
                        execution_status=result.response.status,
                    )

                    await asyncio.to_thread(
                        self.repository.complete_attempt,
                        planned.id,
                        verification,
                    )

                    label = verification.status

                if on_progress is not None:
                    on_progress(
                        f"{planned.task_id} | "
                        f"{planned.model_id} | "
                        f"{planned.effort} | "
                        f"repeat={planned.repeat_index} | "
                        f"{label}"
                    )

            # ----------------------------------
            # 4. Determine final run status
            # ----------------------------------

            final_attempts = await asyncio.to_thread(
                self.repository.list_attempts,
                run_id,
            )

            if self._has_unknown_costs(final_attempts):
                status = "needs_review"

            elif any(a.status == "pending" for a in final_attempts):
                status = "paused_budget"

            elif any(a.status == "error" for a in final_attempts):
                status = "completed_with_errors"

            else:
                status = "completed"

            await asyncio.to_thread(
                self.repository.update_run_status,
                run_id,
                status,
            )

            return self._summary(
                run_id,
                status,
                final_attempts,
            )

        except BaseException:
            await asyncio.to_thread(
                self.repository.update_run_status,
                run_id,
                "interrupted",
            )
            raise
