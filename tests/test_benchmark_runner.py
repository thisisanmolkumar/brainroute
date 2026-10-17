import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from backend.benchmarks.loader import load_tasks
from backend.benchmarks.runner import BenchmarkRunner, ModelEffort
from backend.core.model_registry import ModelRegistry
from backend.providers.exceptions import ProviderTimeoutError
from backend.schemas.execution import ExecutionResult
from backend.schemas.llm import LLMResponse
from backend.services.metrics import ExecutionMetrics

DATASET = "datasets/samples/smoke.jsonl"


class FakeRepository:
    def __init__(self):
        self.run = None
        self.attempts = {}
        self.next_id = 0

    def create_run(self, dataset_path, dataset_sha256):
        self.run = {
            "id": 7,
            "dataset_path": dataset_path,
            "dataset_sha256": dataset_sha256,
            "status": "running",
        }
        return 7

    def create_attempt(
        self,
        run_id,
        task_id,
        model_id,
        effort,
    ):
        self.next_id += 1

        attempt_id = self.next_id

        self.attempts[attempt_id] = {
            "run_id": run_id,
            "task_id": task_id,
            "model_id": model_id,
            "effort": effort,
            "execution_id": None,
            "status": "pending",
        }

        return attempt_id

    def attach_execution(self, attempt_id, execution_id):
        self.attempts[attempt_id]["execution_id"] = execution_id
        self.attempts[attempt_id]["status"] = "running"

    def complete_attempt(self, attempt_id, verification):
        attempt = self.attempts[attempt_id]

        attempt["status"] = "completed"
        attempt["verification_status"] = verification.status
        attempt["correct"] = verification.correct

    def fail_attempt(self, attempt_id, error):
        attempt = self.attempts[attempt_id]

        attempt["status"] = "error"
        attempt["error_type"] = type(error).__name__

    def finish_run(self, run_id, status):
        assert run_id == 7
        self.run["status"] = status


class FakeExecutionService:
    def __init__(self, fail_on_boxes=False):
        self.provider = SimpleNamespace(name="openai")
        self.calls = []

        self.fail_on_boxes = fail_on_boxes

        self.active = 0
        self.max_active = 0

    async def execute(
        self,
        request,
        category=None,
        on_created=None,
    ):
        self.calls.append(request)

        execution_id = 100 + len(self.calls)

        if on_created is not None:
            await on_created(execution_id)

        self.active += 1
        self.max_active = max(
            self.max_active,
            self.active,
        )

        try:
            # Simulate network latency.
            await asyncio.sleep(0.01)

            if self.fail_on_boxes and "8 boxes" in request.prompt:
                raise ProviderTimeoutError("Simulated timeout")

            if "17 multiplied" in request.prompt:
                output = "FINAL_ANSWER: 408"

            elif "8 boxes" in request.prompt:
                output = "FINAL_ANSWER: 42"

            else:
                output = "FINAL_ANSWER: yes"

            response = LLMResponse(
                provider="openai",
                model_id=request.model_id,
                effort=request.effort,
                output_text=output,
                status="completed",
                latency_ms=10,
            )

            metrics = ExecutionMetrics(
                provider="openai",
                model_id=request.model_id,
                effort=request.effort,
                status="completed",
                provider_latency_ms=10,
                end_to_end_latency_ms=12,
                overhead_ms=2,
                usage=None,
                cost=None,
            )

            return ExecutionResult(
                execution_id=execution_id,
                response=response,
                metrics=metrics,
            )

        finally:
            self.active -= 1


@pytest.fixture
def setup_runner():
    repository = FakeRepository()
    service = FakeExecutionService()
    registry = ModelRegistry()

    runner = BenchmarkRunner(
        execution_service=service,
        registry=registry,
        repository=repository,
    )

    return runner, repository, service


def test_benchmark_execution(setup_runner):
    runner, repository, service = setup_runner

    tasks = load_tasks(DATASET)

    result = asyncio.run(
        runner.run(
            tasks=tasks,
            configurations=[
                ModelEffort("gpt-6-luna", "medium"),
            ],
            dataset_path=DATASET,
            max_calls=3,
        )
    )

    assert result.run_id == 7
    assert len(result.outcomes) == 3
    assert len(service.calls) == 3

    assert repository.run["status"] == "completed"

    assert [outcome.verification_status for outcome in result.outcomes] == [
        "correct",
        "incorrect",
        "correct",
    ]


def test_execution_ids_linked(setup_runner):
    runner, repository, _ = setup_runner

    asyncio.run(
        runner.run(
            tasks=load_tasks(DATASET),
            configurations=[
                ModelEffort("gpt-6-luna", "medium"),
            ],
            dataset_path=DATASET,
            max_calls=3,
        )
    )

    assert len(repository.attempts) == 3

    for attempt in repository.attempts.values():
        assert attempt["execution_id"] is not None
        assert attempt["status"] == "completed"


def test_failure_is_recorded():
    repository = FakeRepository()
    service = FakeExecutionService(fail_on_boxes=True)

    runner = BenchmarkRunner(
        execution_service=service,
        registry=ModelRegistry(),
        repository=repository,
    )

    result = asyncio.run(
        runner.run(
            tasks=load_tasks(DATASET),
            configurations=[
                ModelEffort("gpt-6-luna", "medium"),
            ],
            dataset_path=DATASET,
            max_calls=3,
        )
    )

    assert len(result.outcomes) == 3

    errors = [outcome for outcome in result.outcomes if outcome.status == "error"]

    assert len(errors) == 1
    assert errors[0].error_type == "ProviderTimeoutError"
    assert errors[0].execution_id is not None

    assert repository.run["status"] == "completed_with_errors"


def test_max_calls_enforced(setup_runner):
    runner, repository, service = setup_runner

    with pytest.raises(ValueError, match="max_calls"):
        asyncio.run(
            runner.run(
                tasks=load_tasks(DATASET),
                configurations=[
                    ModelEffort("gpt-6-luna", "medium"),
                    ModelEffort("gpt-6-luna", "high"),
                ],
                dataset_path=DATASET,
                max_calls=3,
            )
        )

    assert repository.run is None
    assert len(service.calls) == 0


def test_budget_enforced(setup_runner):
    runner, repository, service = setup_runner

    with pytest.raises(ValueError, match="budget"):
        asyncio.run(
            runner.run(
                tasks=load_tasks(DATASET),
                configurations=[
                    ModelEffort("gpt-6-luna", "medium"),
                ],
                dataset_path=DATASET,
                max_calls=3,
                max_estimated_cost_usd=Decimal("0.000000001"),
            )
        )

    assert repository.run is None
    assert len(service.calls) == 0


def test_bounded_concurrency(setup_runner):
    runner, _, service = setup_runner

    asyncio.run(
        runner.run(
            tasks=load_tasks(DATASET),
            configurations=[
                ModelEffort("gpt-6-luna", "medium"),
                ModelEffort("gpt-6-luna", "high"),
            ],
            dataset_path=DATASET,
            max_calls=6,
            concurrency=2,
        )
    )

    assert len(service.calls) == 6

    assert 1 <= service.max_active <= 2


def test_invalid_configuration(setup_runner):
    runner, repository, service = setup_runner

    with pytest.raises(ValueError):
        asyncio.run(
            runner.run(
                tasks=load_tasks(DATASET),
                configurations=[
                    ModelEffort("gpt-6-luna", "none"),
                ],
                dataset_path=DATASET,
                max_calls=3,
            )
        )

    assert repository.run is None
    assert len(service.calls) == 0
