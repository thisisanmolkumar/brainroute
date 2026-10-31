from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select, update

from backend.benchmarks.verification import VerificationResult
from backend.db.models import (
    BenchmarkAttempt,
    BenchmarkRun,
    Execution,
)
from backend.db.session import SessionLocal


@dataclass(frozen=True)
class SavedRun:
    id: int
    dataset_sha256: str
    plan: dict | None
    budget_usd: Decimal | None
    status: str


@dataclass(frozen=True)
class SavedAttempt:
    id: int
    task_id: str
    model_id: str
    effort: str
    repeat_index: int

    status: str
    verification_status: str | None
    correct: bool | None
    execution_id: int | None

    provider_status: str | None
    output_text: str | None

    cost_usd: Decimal | None
    latency_ms: int | None

    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None

    error_type: str | None


class CollectionRepository:
    def create_run(
        self,
        dataset_path: str,
        dataset_sha256: str,
        plan: dict,
        budget_usd: Decimal,
        attempts: list[tuple[str, str, str, int]],
    ) -> int:
        """
        Create the run and all planned attempts atomically.
        """

        with SessionLocal.begin() as session:
            run = BenchmarkRun(
                dataset_path=dataset_path,
                dataset_sha256=dataset_sha256,
                plan_json=plan,
                max_total_cost_usd=budget_usd,
                status="running",
            )

            session.add(run)
            session.flush()

            session.add_all(
                [
                    BenchmarkAttempt(
                        run_id=run.id,
                        benchmark_task_id=task_id,
                        model_id=model_id,
                        effort=effort,
                        repeat_index=repeat_index,
                        status="pending",
                    )
                    for task_id, model_id, effort, repeat_index in attempts
                ]
            )

            return run.id

    def get_run(self, run_id: int) -> SavedRun | None:
        with SessionLocal() as session:
            run = session.get(BenchmarkRun, run_id)

            if run is None:
                return None

            return SavedRun(
                id=run.id,
                dataset_sha256=run.dataset_sha256,
                plan=run.plan_json,
                budget_usd=run.max_total_cost_usd,
                status=run.status,
            )

    def find_run_by_v2_plan_id(self, plan_id: str) -> int | None:
        """Avoid accidentally starting the same paid v2 plan twice."""
        with SessionLocal() as session:
            statement = (
                select(BenchmarkRun.id)
                .where(BenchmarkRun.plan_json["v2"]["plan_id"].as_string() == plan_id)
                .order_by(BenchmarkRun.id)
                .limit(1)
            )
            return session.scalar(statement)

    def list_attempts(
        self,
        run_id: int,
    ) -> list[SavedAttempt]:

        with SessionLocal() as session:
            statement = (
                select(BenchmarkAttempt, Execution)
                .outerjoin(
                    Execution,
                    BenchmarkAttempt.execution_id == Execution.id,
                )
                .where(BenchmarkAttempt.run_id == run_id)
                .order_by(BenchmarkAttempt.id)
            )

            rows = session.execute(statement).all()

            return [
                SavedAttempt(
                    id=attempt.id,
                    task_id=attempt.benchmark_task_id,
                    model_id=attempt.model_id,
                    effort=attempt.effort,
                    repeat_index=attempt.repeat_index,
                    status=attempt.status,
                    verification_status=(attempt.verification_status),
                    correct=attempt.correct,
                    execution_id=attempt.execution_id,
                    provider_status=(execution.status if execution else None),
                    output_text=(execution.output_text if execution else None),
                    cost_usd=(execution.cost_usd if execution else None),
                    latency_ms=(execution.latency_ms if execution else None),
                    input_tokens=(execution.input_tokens if execution else None),
                    output_tokens=(execution.output_tokens if execution else None),
                    reasoning_tokens=(
                        execution.reasoning_tokens if execution else None
                    ),
                    error_type=(
                        attempt.error_type
                        or (execution.error_type if execution else None)
                    ),
                )
                for attempt, execution in rows
            ]

    def claim_attempt(self, attempt_id: int) -> bool:
        """
        Atomically change pending -> running.

        Only one caller can successfully claim
        the same pending row.
        """

        with SessionLocal.begin() as session:
            statement = (
                update(BenchmarkAttempt)
                .where(
                    BenchmarkAttempt.id == attempt_id,
                    BenchmarkAttempt.status == "pending",
                )
                .values(status="running")
                .returning(BenchmarkAttempt.id)
            )

            claimed_id = session.scalar(statement)

            return claimed_id is not None

    def attach_execution(
        self,
        attempt_id: int,
        execution_id: int,
    ) -> None:
        with SessionLocal.begin() as session:
            attempt = session.get(
                BenchmarkAttempt,
                attempt_id,
            )

            if attempt is None or attempt.status != "running":
                raise ValueError("Attempt is not running")

            attempt.execution_id = execution_id

    def complete_attempt(
        self,
        attempt_id: int,
        verification: VerificationResult,
    ) -> None:
        with SessionLocal.begin() as session:
            attempt = session.get(
                BenchmarkAttempt,
                attempt_id,
            )

            if attempt is None:
                raise ValueError("Attempt not found")

            attempt.status = "completed"
            attempt.verification_status = verification.status
            attempt.correct = verification.correct
            attempt.predicted_answer = verification.predicted_answer
            attempt.error_type = None
            attempt.finished_at = datetime.now(UTC)

            # Update the existing execution record too.
            if attempt.execution_id is not None:
                execution = session.get(
                    Execution,
                    attempt.execution_id,
                )

                if execution is not None:
                    execution.success = verification.correct

    def fail_attempt(
        self,
        attempt_id: int,
        error_type: str,
    ) -> None:
        with SessionLocal.begin() as session:
            attempt = session.get(
                BenchmarkAttempt,
                attempt_id,
            )

            if attempt is None:
                raise ValueError("Attempt not found")

            attempt.status = "error"
            attempt.verification_status = "skipped"
            attempt.correct = None
            attempt.error_type = error_type
            attempt.finished_at = datetime.now(UTC)

    def mark_uncertain(self, attempt_id: int) -> None:
        with SessionLocal.begin() as session:
            attempt = session.get(
                BenchmarkAttempt,
                attempt_id,
            )

            if attempt is None:
                raise ValueError("Attempt not found")

            attempt.status = "uncertain"
            attempt.error_type = "InterruptedUnknownOutcome"
            attempt.finished_at = datetime.now(UTC)

    def update_run_status(
        self,
        run_id: int,
        status: str,
    ) -> None:
        with SessionLocal.begin() as session:
            run = session.get(BenchmarkRun, run_id)

            if run is None:
                raise ValueError("Run not found")

            run.status = status

            if status in {
                "completed",
                "completed_with_errors",
            }:
                run.finished_at = datetime.now(UTC)

    def increase_budget(
        self,
        run_id: int,
        new_budget: Decimal,
    ) -> None:
        with SessionLocal.begin() as session:
            run = session.get(BenchmarkRun, run_id)

            if run is None:
                raise ValueError("Run not found")

            if run.max_total_cost_usd is None:
                raise ValueError("Legacy run has no budget")

            if new_budget <= run.max_total_cost_usd:
                raise ValueError("New budget must exceed current budget")

            run.max_total_cost_usd = new_budget
