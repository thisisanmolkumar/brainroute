"""Purely fake-provider tests: never call a billable model."""

import asyncio
import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.benchmarks.collection import CollectionRunner
from backend.benchmarks.collection_repository import SavedAttempt, SavedRun
from backend.benchmarks.runner import ModelEffort
from backend.datasets.v2_bridge import (
    export_collection_v2,
    materialize_benchmark_dataset,
    validate_collection_plan,
)
from backend.datasets.v2_collection import build_collection_plan
from backend.datasets.v2_schema import TaskV2
from backend.schemas.llm import LLMResponse


class FakeRegistry:
    def validate_configuration(self, model_id, effort):
        if model_id != "model-a" or effort != "medium":
            raise ValueError("Unsupported config")
        pricing = SimpleNamespace(
            input_per_1m_tokens=Decimal("0.01"),
            cached_input_per_1m_tokens=None,
            cache_write_per_1m_tokens=None,
            output_per_1m_tokens=Decimal("0.02"),
            short_context_max_input_tokens=None,
        )
        return SimpleNamespace(provider="openai", pricing=pricing)


class FakeRepository:
    def __init__(self):
        self.run = None
        self.attempts = {}

    def create_run(self, path, sha, plan, budget, schedule):
        self.run = SavedRun(1, sha, plan, budget, "running")
        for index, (task, model, effort, repeat) in enumerate(schedule, 1):
            self.attempts[index] = SavedAttempt(
                id=index,
                task_id=task,
                model_id=model,
                effort=effort,
                repeat_index=repeat,
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
        return 1

    def get_run(self, run_id):
        return self.run if run_id == 1 else None

    def list_attempts(self, run_id):
        assert run_id == 1
        return list(self.attempts.values())

    def _replace(self, attempt_id, **changes):
        self.attempts[attempt_id] = replace(self.attempts[attempt_id], **changes)

    def claim_attempt(self, attempt_id):
        if self.attempts[attempt_id].status != "pending":
            return False
        self._replace(attempt_id, status="running")
        return True

    def attach_execution(self, attempt_id, execution_id):
        self._replace(attempt_id, execution_id=execution_id)

    def complete_attempt(self, attempt_id, verification):
        self._replace(
            attempt_id,
            status="completed",
            verification_status=verification.status,
            correct=verification.correct,
        )

    def fail_attempt(self, attempt_id, error_type):
        self._replace(attempt_id, status="error", error_type=error_type)

    def mark_uncertain(self, attempt_id):
        self._replace(attempt_id, status="uncertain")

    def update_run_status(self, run_id, status):
        self.run = replace(self.run, status=status)


class FakeService:
    def __init__(self, repository):
        self.provider = SimpleNamespace(name="openai")
        self.repository = repository
        self.calls = 0
        self.timeout_values = []

    async def execute(self, request, category=None, on_created=None):
        self.calls += 1
        self.timeout_values.append(request.timeout_seconds)
        execution_id = 100 + self.calls
        await on_created(execution_id)
        for attempt in self.repository.attempts.values():
            if attempt.execution_id == execution_id:
                self.repository._replace(
                    attempt.id,
                    output_text="FINAL_ANSWER: 408",
                    provider_status="completed",
                    cost_usd=Decimal("0.0001"),
                    latency_ms=125,
                    input_tokens=22,
                    output_tokens=13,
                )
                break
        return SimpleNamespace(
            response=LLMResponse(
                provider="openai",
                model_id=request.model_id,
                effort=request.effort,
                output_text="FINAL_ANSWER: 408",
                status="completed",
                latency_ms=125,
            ),
        )


def create_plan(tmp_path: Path, *, n_tasks: int = 2):
    pool_dir = tmp_path / "compiled"
    pool_dir.mkdir()
    pool = pool_dir / "pool_train.jsonl"
    tasks = [
        TaskV2(
            task_id=f"math/{index}",
            source="gsm8k",
            source_revision="fixture-revision",
            source_split="train",
            domain="math",
            raw_prompt=f"Compute {index} times 24.",
            reference_answer="408",
            verifier="gsm8k_final_number",
            verification_status="ready",
            family_id=f"family/{index}",
        )
        for index in range(n_tasks)
    ]
    pool.write_text(
        "".join(task.model_dump_json() + "\n" for task in tasks),
        encoding="utf-8",
    )
    hash_pool = hashlib.sha256(pool.read_bytes()).hexdigest()
    (pool_dir / "manifest.json").write_text(
        json.dumps({"splits": {"train": {"sha256": hash_pool}}}),
        encoding="utf-8",
    )
    # A real historical ML example, so load_ml_examples verifies the snapshot.
    from backend.benchmarks.ml_dataset import (
        MLExample,
        MLMetadata,
        MLTargets,
        TargetMasks,
        extract_features,
    )

    legacy = tmp_path / "old.jsonl"
    example = MLExample(
        features=extract_features(
            "Legacy example not in this collection.", "model-a", "medium"
        ),
        targets=MLTargets(
            success_rate=1,
            successes=1,
            trials=1,
            cost_observations=0,
            latency_observations=0,
        ),
        masks=TargetMasks(cost=False, latency=False),
        metadata=MLMetadata(
            source="previous",
            task_id="original",
            domain="math",
            family_id="old",
            original_split="train",
            group_key="old",
            run_ids=[1],
        ),
    )
    legacy.write_text(example.model_dump_json() + "\n", encoding="utf-8")
    plan = build_collection_plan(
        tasks,
        candidates=["model-a:medium"],
        legacy_prompts=[],
        max_tasks=n_tasks,
        repeats=2,
        budget_usd="0.50",
        per_attempt_cap_usd="0.005",
        seed=42,
        max_output_tokens=128,
        timeout_seconds=30,
    )
    plan_dir = tmp_path / "collection_plan"
    plan_dir.mkdir()
    attempts = plan_dir / "attempts.jsonl"
    attempts.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in plan["attempts"]),
        encoding="utf-8",
    )
    manifest = {k: v for k, v in plan.items() if k != "attempts"}
    manifest.update(
        {
            "pool_sha256": hash_pool,
            "attempts_sha256": hashlib.sha256(attempts.read_bytes()).hexdigest(),
            "legacy_data_sha256": {
                str(legacy): hashlib.sha256(legacy.read_bytes()).hexdigest(),
            },
        }
    )
    (plan_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return plan_dir, pool, legacy


def test_v2_plan_materialization_preserves_raw_prompt(tmp_path):
    plan_dir, pool, _ = create_plan(tmp_path)
    plan = validate_collection_plan(plan_dir, pool)
    assert len(plan.attempts) == 4
    assert plan.benchmark_tasks[0].metadata["v2_raw_prompt"].startswith("Compute")
    assert "FINAL_ANSWER:" in plan.benchmark_tasks[0].prompt
    assert "FINAL_ANSWER: 408" not in plan.benchmark_tasks[0].prompt
    path = materialize_benchmark_dataset(plan, create=True)
    assert path.is_file()
    assert materialize_benchmark_dataset(plan, create=False) == path


def test_tampered_attempts_fail_before_any_call(tmp_path):
    plan_dir, pool, _ = create_plan(tmp_path)
    path = plan_dir / "attempts.jsonl"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_collection_plan(plan_dir, pool)


def test_tampered_pool_fails(tmp_path):
    plan_dir, pool, _ = create_plan(tmp_path)
    pool.write_text(pool.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_collection_plan(plan_dir, pool)


def test_tampered_history_fails(tmp_path):
    plan_dir, pool, legacy = create_plan(tmp_path)
    legacy.write_text(legacy.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Historical"):
        validate_collection_plan(plan_dir, pool)


def test_immutable_materialization_rejects_edits(tmp_path):
    plan_dir, pool, _ = create_plan(tmp_path)
    plan = validate_collection_plan(plan_dir, pool)
    path = materialize_benchmark_dataset(plan, create=True)
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        materialize_benchmark_dataset(plan, create=False)


def test_integration_collect_export_resume(tmp_path):
    plan_dir, pool, _ = create_plan(tmp_path)
    plan = validate_collection_plan(plan_dir, pool)
    materialize_benchmark_dataset(plan, create=True)
    repo = FakeRepository()
    service = FakeService(repo)
    runner = CollectionRunner(service, FakeRegistry(), repository=repo)

    def collect(resume=None, **changes):
        arguments = dict(
            dataset_path=plan.dataset_path,
            configurations=[ModelEffort(*c) for c in plan.configurations]
            if resume is None
            else None,
            repeats=plan.repeats,
            seed=plan.seed,
            max_calls=len(plan.attempts),
            max_output_tokens=plan.max_output_tokens,
            budget_usd=plan.budget_usd,
            timeout_seconds=plan.timeout_seconds,
            max_estimated_attempt_cost_usd=plan.attempt_cap_usd,
            plan_metadata=plan.metadata,
            resume_run_id=resume,
        )
        arguments.update(changes)
        return asyncio.run(runner.collect(**arguments))

    summary = collect()
    assert summary.status == "completed"
    assert service.calls == 4
    assert service.timeout_values == [30] * 4
    assert summary.known_cost_usd == Decimal("0.0004")

    output = tmp_path / "v2_export.jsonl"
    assert export_collection_v2(plan, summary.run_id, output, repo) == 4
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(row["prompt"].startswith("Compute") for row in rows)
    assert all(row["strict_success"] is True for row in rows)
    assert all(row["cost_usd"] == "0.0001" for row in rows)
    assert export_collection_v2(plan, summary.run_id, output, repo) == 4

    resumed = collect(resume=summary.run_id)
    assert resumed.status == "completed"
    assert service.calls == 4  # no duplicate paid calls

    with pytest.raises(ValueError, match="identity mismatch"):
        collect(resume=summary.run_id, plan_metadata={"plan_id": "wrong"})
    with pytest.raises(ValueError, match="timeout differs"):
        collect(resume=summary.run_id, timeout_seconds=40)


def test_attempt_cap_enforced_before_execution(tmp_path):
    plan_dir, pool, _ = create_plan(tmp_path)
    plan = validate_collection_plan(plan_dir, pool)
    materialize_benchmark_dataset(plan, create=True)
    repo = FakeRepository()
    service = FakeService(repo)
    runner = CollectionRunner(service, FakeRegistry(), repository=repo)
    with pytest.raises(ValueError, match="Estimated attempt cost"):
        asyncio.run(
            runner.collect(
                dataset_path=plan.dataset_path,
                configurations=[ModelEffort(*c) for c in plan.configurations],
                max_calls=4,
                repeats=2,
                max_output_tokens=128,
                budget_usd=Decimal("0.50"),
                max_estimated_attempt_cost_usd=Decimal("0.000000001"),
                plan_metadata=plan.metadata,
                timeout_seconds=30,
            )
        )
    assert service.calls == 0


def test_planning_cli_creates_reusable_manifest(tmp_path, monkeypatch):
    import sys

    from scripts.plan_collection_v2 import main

    _, pool, legacy = create_plan(tmp_path)
    output = tmp_path / "generated_by_cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_collection_v2",
            "--pool",
            str(pool),
            "--legacy-data",
            str(legacy),
            "--candidate",
            "model-a:medium",
            "--max-tasks",
            "2",
            "--repeats",
            "2",
            "--budget-usd",
            "0.50",
            "--per-attempt-cap-usd",
            "0.005",
            "--output-dir",
            str(output),
        ],
    )
    main()
    assert validate_collection_plan(output, pool).plan_id


def test_unknown_provider_cost_is_not_a_failed_task_label(tmp_path):
    plan_dir, pool, _ = create_plan(tmp_path)
    plan = validate_collection_plan(plan_dir, pool)
    materialize_benchmark_dataset(plan, create=True)
    repo = FakeRepository()

    class UncertainService(FakeService):
        async def execute(self, request, category=None, on_created=None):
            self.calls += 1
            await on_created(101)
            raise RuntimeError("Provider connection lost")

    service = UncertainService(repo)
    runner = CollectionRunner(service, FakeRegistry(), repository=repo)
    kwargs = dict(
        dataset_path=plan.dataset_path,
        configurations=[ModelEffort(*c) for c in plan.configurations],
        repeats=plan.repeats,
        max_calls=4,
        max_output_tokens=128,
        budget_usd=plan.budget_usd,
        timeout_seconds=30,
        max_estimated_attempt_cost_usd=plan.attempt_cap_usd,
        plan_metadata=plan.metadata,
    )
    result = asyncio.run(runner.collect(**kwargs))
    assert result.status == "needs_review"
    assert service.calls == 1
    output = tmp_path / "errors.jsonl"
    export_collection_v2(plan, result.run_id, output, repo)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(record["strict_success"] is None for record in records)
    assert all(record["outcome"] == "execution_unverified" for record in records)
    assert all(record["cost_usd"] is None for record in records)

    resume_kwargs = {**kwargs, "resume_run_id": result.run_id, "configurations": None}
    resumed = asyncio.run(runner.collect(**resume_kwargs))
    assert resumed.status == "needs_review"
    assert service.calls == 1  # uncertain provider calls are never retried
