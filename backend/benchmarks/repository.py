from datetime import UTC, datetime

from backend.benchmarks.verification import VerificationResult
from backend.db.models import BenchmarkAttempt, BenchmarkRun
from backend.db.session import SessionLocal


class BenchmarkRepository:
    def create_run(
        self,
        dataset_path: str,
        dataset_sha256: str,
    ) -> int:

        with SessionLocal.begin() as session:
            run = BenchmarkRun(
                dataset_path=dataset_path,
                dataset_sha256=dataset_sha256,
                status="running",
            )

            session.add(run)
            session.flush()

            return run.id

    def create_attempt(
        self,
        run_id: int,
        task_id: str,
        model_id: str,
        effort: str,
    ) -> int:

        with SessionLocal.begin() as session:
            attempt = BenchmarkAttempt(
                run_id=run_id,
                benchmark_task_id=task_id,
                model_id=model_id,
                effort=effort,
                status="pending",
            )

            session.add(attempt)
            session.flush()

            return attempt.id

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

            if attempt is None:
                raise ValueError(f"Benchmark attempt {attempt_id} not found")

            attempt.execution_id = execution_id
            attempt.status = "running"

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
                raise ValueError(f"Benchmark attempt {attempt_id} not found")

            attempt.status = "completed"
            attempt.verification_status = verification.status

            attempt.correct = verification.correct
            attempt.predicted_answer = verification.predicted_answer

            attempt.finished_at = datetime.now(UTC)

    def fail_attempt(
        self,
        attempt_id: int,
        error: Exception,
    ) -> None:

        with SessionLocal.begin() as session:
            attempt = session.get(
                BenchmarkAttempt,
                attempt_id,
            )

            if attempt is None:
                raise ValueError(f"Benchmark attempt {attempt_id} not found")

            attempt.status = "error"
            attempt.error_type = type(error).__name__

            attempt.finished_at = datetime.now(UTC)

    def finish_run(
        self,
        run_id: int,
        status: str,
    ) -> None:

        with SessionLocal.begin() as session:
            run = session.get(BenchmarkRun, run_id)

            if run is None:
                raise ValueError(f"Benchmark run {run_id} not found")

            run.status = status
            run.finished_at = datetime.now(UTC)
