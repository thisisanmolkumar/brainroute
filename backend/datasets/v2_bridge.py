"""Validated, no-billing bridge from immutable v2 plans to the Phase 2 collector.

The Phase 2 collector remains the only component that sends provider requests.
No TaskV2 source is executable until these cross-file integrity checks pass.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from backend.benchmarks.schema import BenchmarkTask
from backend.datasets.v2_builder import content_hash, load_tasks
from backend.datasets.v2_collection import (
    PROTOCOL_VERSION,
    overlaps_legacy,
    stable_hash,
)
from backend.datasets.v2_readiness import assert_collectible, render_execution_prompt
from backend.datasets.v2_schema import TaskV2
from backend.ml.data import load_ml_examples

VERIFIER_MAP = {
    "gsm8k_final_number": "numeric_final",
    "bbh_boolean": "exact_final",
}
DOMAIN_MAP = {"math": "math", "logic": "logic"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_id(task: TaskV2) -> str:
    """Stable, DB-safe identifier, independent of the external task ID syntax."""
    identity = {"source": task.source, "task_id": task.task_id}
    return "v2_" + stable_hash(identity)[:48]


@dataclass(frozen=True)
class ValidatedCollectionPlan:
    plan_dir: Path
    pool_path: Path
    manifest: dict
    attempts: list[dict]
    tasks: list[TaskV2]
    benchmark_tasks: list[BenchmarkTask]
    configurations: list[tuple[str, str]]
    repeats: int
    budget_usd: Decimal
    attempt_cap_usd: Decimal
    max_output_tokens: int
    timeout_seconds: int
    seed: int

    @property
    def plan_id(self) -> str:
        return self.manifest["plan_id"]

    @property
    def dataset_path(self) -> Path:
        return self.plan_dir / "benchmark_tasks.jsonl"

    @property
    def metadata(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "pool_sha256": self.manifest["pool_sha256"],
            "attempts_sha256": self.manifest["attempts_sha256"],
            "protocol_version": PROTOCOL_VERSION,
        }


def validate_collection_plan(
    plan_dir: Path, pool_path: Path
) -> ValidatedCollectionPlan:
    """Fail closed on edits, split leakage, missing verifiers, or mismatched schedules."""
    plan_dir = Path(plan_dir)
    pool_path = Path(pool_path)
    if pool_path.name != "pool_train.jsonl":
        raise ValueError("A v2 training run requires pool_train.jsonl")

    manifest = json.loads((plan_dir / "manifest.json").read_text(encoding="utf-8"))
    attempts_path = plan_dir / "attempts.jsonl"
    if sha256_file(attempts_path) != manifest["attempts_sha256"]:
        raise ValueError("Planned attempts checksum mismatch")
    if sha256_file(pool_path) != manifest["pool_sha256"]:
        raise ValueError("Pool checksum mismatch against collection plan")
    compiled = json.loads(
        (pool_path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if compiled["splits"]["train"]["sha256"] != sha256_file(pool_path):
        raise ValueError("Pool checksum mismatch against compiled manifest")

    attempts = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not attempts:
        raise ValueError("Empty collection plan")

    # Reconstruct the exact object used by the planner for its immutable ID.
    original_plan = {
        key: value
        for key, value in manifest.items()
        if key
        not in {"plan_id", "pool_sha256", "legacy_data_sha256", "attempts_sha256"}
    }
    original_plan["attempts"] = attempts
    if stable_hash(original_plan) != manifest["plan_id"]:
        raise ValueError("Plan identity mismatch")

    if (
        manifest["protocol_version"] != PROTOCOL_VERSION
        or manifest["pool_split"] != "train"
        or manifest["execution_enabled"] is not False
        or manifest["max_inflight"] != 1
    ):
        raise ValueError("Unsupported or unsafe collection protocol")

    configs = [(item["model_id"], item["effort"]) for item in manifest["candidates"]]
    if not configs or len(configs) != len(set(configs)):
        raise ValueError("Invalid candidate configurations")

    if manifest["selected_tasks"] <= 0:
        raise ValueError("Empty or invalid selected task count")
    repeats = manifest["planned_attempts"] // (
        len(configs) * manifest["selected_tasks"]
    )
    if not 1 <= repeats <= 10:
        raise ValueError("Invalid repeat count")
    if repeats * len(configs) * manifest["selected_tasks"] != len(attempts):
        raise ValueError("Incomplete or irregular candidate coverage")
    max_output_tokens = manifest["max_output_tokens"]
    timeout_seconds = manifest["timeout_seconds"]
    if not 16 <= max_output_tokens <= 4096 or not 0 < timeout_seconds <= 120:
        raise ValueError("Invalid execution limits")
    attempt_cap = Decimal(manifest["per_attempt_cap_usd"])
    budget = Decimal(manifest["budget_usd"])
    if (
        not attempt_cap.is_finite()
        or not budget.is_finite()
        or attempt_cap <= 0
        or budget <= 0
        or attempt_cap > budget
    ):
        raise ValueError("Invalid planned budget")

    # Historical provenance must still exist and match the planning snapshot.
    legacy_prompts = []
    legacy_hashes = manifest["legacy_data_sha256"]
    if not legacy_hashes:
        raise ValueError("Missing Dataset v1 leakage reference")
    for saved_path, expected_hash in legacy_hashes.items():
        path = Path(saved_path)
        if sha256_file(path) != expected_hash:
            raise ValueError("Historical Dataset v1 checksum mismatch")
        legacy_prompts.extend(
            example.features.prompt for example in load_ml_examples(path)
        )

    pool = load_tasks([pool_path])
    tasks_by_identity = {(item.source, item.task_id): item for item in pool}
    if len(tasks_by_identity) != len(pool):
        raise ValueError("Duplicate pool identities")

    selected: dict[tuple[str, str], TaskV2] = {}
    counts: Counter[tuple[str, str, str, str]] = Counter()
    attempt_ids: set[str] = set()

    for attempt in attempts:
        identity = (attempt["source"], attempt["task_id"])
        task = tasks_by_identity.get(identity)
        if task is None:
            raise ValueError(f"Attempt references missing task: {identity}")
        if task.source_split != "train":
            raise ValueError("Held-out tasks cannot enter training collection")
        assert_collectible(task)
        if task.verifier not in VERIFIER_MAP or task.domain not in DOMAIN_MAP:
            raise ValueError("Task verifier or domain unsupported by safe bridge")
        if overlaps_legacy(task.raw_prompt, legacy_prompts):
            raise ValueError("Dataset v1 overlap discovered during execution preflight")

        rendered = render_execution_prompt(task)
        execution_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        raw_hash = content_hash(task.raw_prompt)
        if (
            attempt["source_revision"] != task.source_revision
            or attempt["domain"] != task.domain
            or attempt["family_id"] != task.family_id
            or attempt["verifier"] != task.verifier
            or attempt["raw_prompt_sha256"] != raw_hash
            or attempt["execution_prompt"] != rendered
            or attempt["execution_prompt_sha256"] != execution_hash
        ):
            raise ValueError("Attempt task identity or execution prompt mismatch")
        if (
            attempt["max_output_tokens"] != max_output_tokens
            or attempt["timeout_seconds"] != timeout_seconds
            or Decimal(attempt["per_attempt_cap_usd"]) != attempt_cap
        ):
            raise ValueError("Attempt limits differ from the experiment plan")

        config = (attempt["model_id"], attempt["effort"])
        repeat = attempt["repeat_index"]
        if (
            config not in configs
            or type(repeat) is not int
            or not 0 <= repeat < repeats
        ):
            raise ValueError("Unexpected model-effort or repeat in plan")
        attempt_identity = {
            "protocol": PROTOCOL_VERSION,
            "source": task.source,
            "source_revision": task.source_revision,
            "task_id": task.task_id,
            "raw_prompt_sha256": raw_hash,
            "verifier": task.verifier,
            "model_id": config[0],
            "effort": config[1],
            "repeat_index": repeat,
            "execution_prompt_sha256": execution_hash,
        }
        if stable_hash(attempt_identity) != attempt["attempt_id"]:
            raise ValueError("Attempt ID mismatch")
        if attempt["attempt_id"] in attempt_ids:
            raise ValueError("Duplicate attempt ID")
        attempt_ids.add(attempt["attempt_id"])
        selected[identity] = task
        counts[(*identity, *config)] += 1

    if len(selected) != manifest["selected_tasks"]:
        raise ValueError("Selected task count mismatch")
    if any(count != repeats for count in counts.values()):
        raise ValueError("Unbalanced configuration repetitions")
    if len(counts) != len(selected) * len(configs):
        raise ValueError("Missing model-effort comparisons")
    if len({(t.source, t.family_id) for t in selected.values()}) != len(selected):
        raise ValueError("Multiple variants from one task family")
    if len({content_hash(t.raw_prompt) for t in selected.values()}) != len(selected):
        raise ValueError("Duplicate prompt selected")

    benchmark_tasks = []
    for task in sorted(selected.values(), key=lambda item: (item.source, item.task_id)):
        # Only an execution prompt is ever sent to the provider. Raw text is
        # retained as metadata for the v2 ML exporter and auditing.
        benchmark_tasks.append(
            BenchmarkTask(
                task_id=benchmark_id(task),
                domain=DOMAIN_MAP[task.domain],
                source=task.source,
                split="train",
                prompt=render_execution_prompt(task),
                reference_answer=task.reference_answer,
                verifier=VERIFIER_MAP[task.verifier],
                family_id=f"{task.source}:{task.family_id}",
                metadata={
                    "v2_plan_id": manifest["plan_id"],
                    "v2_raw_prompt": task.raw_prompt,
                    "v2_task_id": task.task_id,
                    "v2_source_revision": task.source_revision,
                    "v2_verifier": task.verifier,
                    "v2_raw_prompt_sha256": content_hash(task.raw_prompt),
                },
            )
        )

    # Pydantic's legacy BenchmarkTask schema strips outer whitespace.
    # Reject rather than silently changing the planned execution prompt.
    ordered_tasks = sorted(
        selected.values(), key=lambda task: (task.source, task.task_id)
    )
    for bench, task in zip(benchmark_tasks, ordered_tasks, strict=True):
        if bench.prompt != render_execution_prompt(task):
            raise ValueError("Legacy benchmark schema altered the execution prompt")

    return ValidatedCollectionPlan(
        plan_dir=plan_dir,
        pool_path=pool_path,
        manifest=manifest,
        attempts=attempts,
        tasks=list(selected.values()),
        benchmark_tasks=benchmark_tasks,
        configurations=configs,
        repeats=repeats,
        budget_usd=budget,
        attempt_cap_usd=attempt_cap,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        seed=manifest["seed"],
    )


def materialize_benchmark_dataset(
    plan: ValidatedCollectionPlan, *, create: bool
) -> Path:
    """Create once, then require the exact same bytes for every resume."""
    output = plan.dataset_path
    data = "".join(task.model_dump_json() + "\n" for task in plan.benchmark_tasks)
    if output.exists():
        if output.read_text(encoding="utf-8") != data:
            raise ValueError("Immutable benchmark dataset was modified")
    elif create:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(data)
    else:
        raise FileNotFoundError("Cannot resume without original benchmark dataset")
    return output


def export_collection_v2(
    plan: ValidatedCollectionPlan,
    run_id: int,
    output_path: Path,
    repository: object,
) -> int:
    """Export v2 observations with raw task features and nullable success labels."""
    from backend.datasets.v2_outcomes import classify_observation

    run = repository.get_run(run_id)
    if run is None or run.dataset_sha256 != sha256_file(plan.dataset_path):
        raise ValueError("Collection run or dataset checksum mismatch")
    if run.plan is None or run.plan.get("v2") != plan.metadata:
        raise ValueError("Collection run does not match the v2 plan")

    task_map = {benchmark_id(task): task for task in plan.tasks}
    source_tasks = {(task.source, task.task_id): task for task in plan.tasks}
    attempt_map = {
        (
            benchmark_id(source_tasks[(item["source"], item["task_id"])]),
            item["model_id"],
            item["effort"],
            item["repeat_index"],
        ): item["attempt_id"]
        for item in plan.attempts
    }
    saved_attempts = repository.list_attempts(run_id)
    if len(saved_attempts) != len(plan.attempts):
        raise ValueError("Stored attempt count differs from planned coverage")

    records = []
    for attempt in saved_attempts:
        task = task_map[attempt.task_id]
        lookup_key = (
            attempt.task_id,
            attempt.model_id,
            attempt.effort,
            attempt.repeat_index,
        )
        if lookup_key not in attempt_map:
            raise ValueError("Unexpected attempt in persisted collection")
        if attempt.status == "completed":
            outcome = classify_observation(
                task,
                provider_status=attempt.provider_status or "unknown",
                response_text=attempt.output_text,
            )
        else:
            outcome = {
                "outcome": "execution_unverified",
                "success": None,
                "reason": attempt.error_type or attempt.status,
            }
        records.append(
            {
                "run_id": run_id,
                "plan_id": plan.plan_id,
                "attempt_id": attempt_map[lookup_key],
                "task_id": task.task_id,
                "source_revision": task.source_revision,
                "source": task.source,
                "domain": task.domain,
                "family_id": task.family_id,
                "split": "train",
                "prompt": task.raw_prompt,
                "execution_prompt_sha256": hashlib.sha256(
                    render_execution_prompt(task).encode("utf-8")
                ).hexdigest(),
                "reference_answer": task.reference_answer,
                "verifier": task.verifier,
                "model_id": attempt.model_id,
                "effort": attempt.effort,
                "repeat_index": attempt.repeat_index,
                "attempt_status": attempt.status,
                "provider_status": attempt.provider_status,
                "verification_status": attempt.verification_status,
                "outcome": outcome["outcome"],
                "outcome_reason": outcome.get("reason"),
                "strict_success": outcome["success"],
                "execution_id": attempt.execution_id,
                "cost_usd": str(attempt.cost_usd)
                if attempt.cost_usd is not None
                else None,
                "latency_ms": attempt.latency_ms,
                "input_tokens": attempt.input_tokens,
                "output_tokens": attempt.output_tokens,
                "reasoning_tokens": attempt.reasoning_tokens,
                "error_type": attempt.error_type,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != data:
            raise ValueError(
                "Existing v2 export differs from current saved observations"
            )
    else:
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(data)
    return len(records)
