import asyncio
import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.benchmarks.collection import (
    CollectionRunner,
    build_schedule,
)
from backend.benchmarks.collection_repository import (
    SavedAttempt,
    SavedRun,
)
from backend.benchmarks.exporter import export_collection
from backend.benchmarks.loader import load_tasks
from backend.benchmarks.runner import ModelEffort
from backend.core.model_registry import ModelRegistry
from backend.schemas.llm import LLMResponse

DATASET = Path("datasets/samples/smoke.jsonl")


class FakeRepository:
    def __init__(self):
        self.run = None
        self.attempts = {}
        self.next_id = 1

    def create_run(
        self,
        dataset_path,
        dataset_sha256,
        plan,
        budget_usd,
        attempts,
    ):
        self.run = SavedRun(
            id=7,
            dataset_sha256=dataset_sha256,
            plan=plan,
            budget_usd=budget_usd,
            status="running",
        )

        for task_id, model_id, effort, repeat_index in attempts:
            attempt_id = self.next_id
            self.next_id += 1

            self.attempts[attempt_id] = SavedAttempt(
                id=attempt_id,
                task_id=task_id,
                model_id=model_id,
                effort=effort,
                repeat_index=repeat_index,
                status="pending",
                verification_status=None,
                correct=None,
                execution_id=None,
                provider_status=None,
                output_text=None,
                cost_usd=None,
                latency_ms=None,
                input_tokens=None,
                output_tokens=None,
                reasoning_tokens=None,
                error_type=None,
            )

        return 7

    def get_run(self, run_id):
        if self.run and self.run.id == run_id:
            return self.run
        return None

    def list_attempts(self, run_id):
        assert run_id == 7
        return list(self.attempts.values())

    def _update(self, attempt_id, **changes):
        self.attempts[attempt_id] = replace(
            self.attempts[attempt_id],
            **changes,
        )

    def claim_attempt(self, attempt_id):
        if self.attempts[attempt_id].status != "pending":
            return False

        self._update(attempt_id, status="running")
        return True

    def attach_execution(self, attempt_id, execution_id):
        self._update(
            attempt_id,
            execution_id=execution_id,
        )

    def save_provider_response(
        self,
        execution_id,
        output_text,
    ):
        for attempt in self.attempts.values():
            if attempt.execution_id == execution_id:
                self._update(
                    attempt.id,
                    provider_status="completed",
                    output_text=output_text,
                    cost_usd=Decimal("0.0001"),
                    latency_ms=100,
                    input_tokens=30,
                    output_tokens=50,
                    reasoning_tokens=20,
                )
                return

        raise ValueError("Unknown execution")

    def complete_attempt(self, attempt_id, verification):
        self._update(
            attempt_id,
            status="completed",
            verification_status=verification.status,
            correct=verification.correct,
            error_type=None,
        )

    def fail_attempt(self, attempt_id, error_type):
        self._update(
            attempt_id,
            status="error",
            verification_status="skipped",
            error_type=error_type,
        )

    def mark_uncertain(self, attempt_id):
        self._update(
            attempt_id,
            status="uncertain",
            error_type="InterruptedUnknownOutcome",
        )

    def update_run_status(self, run_id, status):
        assert run_id == 7
        self.run = replace(self.run, status=status)

    def increase_budget(self, run_id, new_budget):
        assert run_id == 7

        if new_budget <= self.run.budget_usd:
            raise ValueError("Budget must increase")

        self.run = replace(
            self.run,
            budget_usd=new_budget,
        )


class FakeExecutionService:
    def __init__(self, repository):
        self.repository = repository
        self.provider = SimpleNamespace(name="openai")
        self.calls = 0

    async def execute(
        self,
        request,
        category=None,
        on_created=None,
    ):
        self.calls += 1
        execution_id = self.calls + 100

        if on_created is not None:
            await on_created(execution_id)

        if "17 multiplied" in request.prompt:
            output = "FINAL_ANSWER: 408"
        elif "8 boxes" in request.prompt:
            output = "FINAL_ANSWER: 43"
        else:
            output = "FINAL_ANSWER: yes"

        self.repository.save_provider_response(
            execution_id,
            output,
        )

        return SimpleNamespace(
            execution_id=execution_id,
            response=LLMResponse(
                provider="openai",
                model_id=request.model_id,
                effort=request.effort,
                output_text=output,
                status="completed",
                latency_ms=100,
            ),
        )


@pytest.fixture
def setup_collection():
    repository = FakeRepository()
    service = FakeExecutionService(repository)

    runner = CollectionRunner(
        execution_service=service,
        registry=ModelRegistry(),
        repository=repository,
    )

    return runner, repository, service


def run_collection(runner, **kwargs):
    return asyncio.run(
        runner.collect(
            dataset_path=DATASET,
            configurations=[
                ModelEffort("gpt-6-luna", "medium"),
            ],
            max_calls=20,
            **kwargs,
        )
    )


def test_repeat_schedule():
    tasks = load_tasks(DATASET)

    configurations = [
        ModelEffort("gpt-6-luna", "medium"),
        ModelEffort("gpt-6-luna", "high"),
    ]

    schedule = build_schedule(
        tasks,
        configurations,
        repeats=3,
        seed=42,
    )

    assert len(schedule) == 18
    assert len(set(schedule)) == 18

    # Same seed -> same execution order.
    assert schedule == build_schedule(
        tasks,
        configurations,
        repeats=3,
        seed=42,
    )


def test_repeated_collection(setup_collection):
    runner, repository, service = setup_collection

    summary = run_collection(
        runner,
        repeats=2,
    )

    assert summary.total == 6
    assert summary.completed == 6
    assert summary.status == "completed"

    assert service.calls == 6

    assert (
        len(
            {
                (
                    a.task_id,
                    a.model_id,
                    a.effort,
                    a.repeat_index,
                )
                for a in repository.attempts.values()
            }
        )
        == 6
    )


def test_resume_does_not_repeat_completed_work(
    setup_collection,
):
    runner, _, service = setup_collection

    first = run_collection(
        runner,
        repeats=1,
    )

    assert first.status == "completed"
    assert service.calls == 3

    second = run_collection(
        runner,
        resume_run_id=first.run_id,
    )

    assert second.status == "completed"
    assert service.calls == 3


def test_insufficient_budget_pauses(
    setup_collection,
):
    runner, _, service = setup_collection

    summary = run_collection(
        runner,
        repeats=1,
        budget_usd=Decimal("0.000000001"),
    )

    assert summary.status == "paused_budget"
    assert summary.pending == 3
    assert service.calls == 0


def test_changed_dataset_cannot_resume(
    setup_collection,
    tmp_path,
):
    runner, _, _ = setup_collection

    first = run_collection(
        runner,
        repeats=1,
    )

    changed = tmp_path / "changed.jsonl"

    original = DATASET.read_text(encoding="utf-8")

    changed.write_text(
        original + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="checksum mismatch",
    ):
        asyncio.run(
            runner.collect(
                dataset_path=changed,
                resume_run_id=first.run_id,
            )
        )


def test_recover_saved_execution(setup_collection):
    runner, repository, service = setup_collection

    tasks = load_tasks(DATASET)[:1]

    plan = {
        "version": 1,
        "seed": 42,
        "repeats": 1,
        "max_output_tokens": 1024,
        "task_ids": [tasks[0].task_id],
        "configurations": [
            {
                "model_id": "gpt-6-luna",
                "effort": "medium",
            }
        ],
    }

    sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()

    repository.create_run(
        str(DATASET),
        sha,
        plan,
        Decimal("0.10"),
        [(tasks[0].task_id, "gpt-6-luna", "medium", 0)],
    )

    # Simulate a provider response already saved
    # before the collector was interrupted.
    repository._update(
        1,
        status="running",
        execution_id=101,
        provider_status="completed",
        output_text="FINAL_ANSWER: 408",
        cost_usd=Decimal("0.0001"),
    )

    # The saved plan used only one task.
    # Use a matching temporary JSONL file.
    #
    # This test directly invokes recovery because
    # the original full dataset has three tasks.
    asyncio.run(
        runner._recover(
            7,
            {tasks[0].task_id: tasks[0]},
        )
    )

    assert repository.attempts[1].status == "completed"
    assert repository.attempts[1].correct is True
    assert service.calls == 0


def test_uncertain_attempt_is_not_retried(
    setup_collection,
):
    runner, repository, service = setup_collection

    summary = run_collection(
        runner,
        repeats=1,
        budget_usd=Decimal("0.000000001"),
    )

    assert summary.status == "paused_budget"

    # Simulate interruption after an attempt was
    # claimed, with no confirmed provider response.
    repository._update(
        1,
        status="running",
    )

    resumed = run_collection(
        runner,
        resume_run_id=summary.run_id,
    )

    assert resumed.status == "needs_review"
    assert resumed.uncertain == 1
    assert service.calls == 0


def test_export_collection(
    setup_collection,
    tmp_path,
):
    runner, repository, _ = setup_collection

    summary = run_collection(
        runner,
        repeats=1,
    )

    output = tmp_path / "training.jsonl"

    count = export_collection(
        run_id=summary.run_id,
        dataset_path=DATASET,
        output_path=output,
        repository=repository,
    )

    assert count == 3

    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 3

    assert all(row["strict_success"] is True for row in rows)

    assert all(row["cost_usd"] is not None for row in rows)


def test_resume_after_budget_increase(setup_collection):
    runner, _, service = setup_collection

    paused = run_collection(
        runner,
        repeats=1,
        budget_usd=Decimal("0.000000001"),
    )

    assert paused.status == "paused_budget"
    assert service.calls == 0

    resumed = run_collection(
        runner,
        resume_run_id=paused.run_id,
        increase_budget_to=Decimal("0.10"),
    )

    assert resumed.status == "completed"
    assert resumed.completed == 3
    assert service.calls == 3
