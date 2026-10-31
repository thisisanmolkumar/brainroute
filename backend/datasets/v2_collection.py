import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation

from backend.datasets.v2_builder import (
    content_hash,
    normalize_prompt,
)
from backend.datasets.v2_readiness import (
    assert_collectible,
    render_execution_prompt,
)
from backend.datasets.v2_schema import TaskV2

PROTOCOL_VERSION = "v2-pilot-1"
ALLOWED_EFFORTS = {"low", "medium", "high", "xhigh"}


def stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def parse_candidate(spec: str) -> dict[str, str]:
    model_id, separator, effort = spec.rpartition(":")

    if not separator or not model_id.strip() or effort not in ALLOWED_EFFORTS:
        raise ValueError("Candidate must be model_id:effort")

    return {
        "model_id": model_id.strip(),
        "effort": effort,
    }


def parse_positive_amount(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid spending amount") from exc

    if not amount.is_finite() or amount <= 0:
        raise ValueError("Spending amount must be positive and finite")

    return amount


def overlaps_legacy(
    raw_prompt: str,
    legacy_prompts: list[str],
) -> bool:
    current = normalize_prompt(raw_prompt)

    for previous in legacy_prompts:
        previous = normalize_prompt(previous)

        if current == previous:
            return True

        # Dataset v1 may contain a wrapper around
        # the original benchmark question.
        if len(current) >= 20 and current in previous:
            return True

    return False


def build_collection_plan(
    tasks: list[TaskV2],
    *,
    candidates: list[str],
    legacy_prompts: list[str],
    max_tasks: int,
    repeats: int,
    budget_usd: str,
    per_attempt_cap_usd: str,
    seed: int = 42,
    max_output_tokens: int = 256,
    timeout_seconds: int = 30,
) -> dict:
    if max_tasks < 1 or repeats < 1:
        raise ValueError("max_tasks and repeats must be positive")

    if max_output_tokens < 1 or timeout_seconds < 1:
        raise ValueError("Invalid execution limits")

    budget = parse_positive_amount(budget_usd)
    attempt_cap = parse_positive_amount(per_attempt_cap_usd)

    if attempt_cap > budget:
        raise ValueError("Per-attempt cap exceeds total budget")

    configurations = [parse_candidate(spec) for spec in candidates]

    signatures = [(item["model_id"], item["effort"]) for item in configurations]

    if not signatures or len(set(signatures)) != len(signatures):
        raise ValueError("Candidates must be present and unique")

    configurations.sort(
        key=lambda item: (
            item["model_id"],
            item["effort"],
        )
    )

    # Stable ordering independent of input JSONL order.
    ordered = sorted(
        tasks,
        key=lambda task: stable_hash(
            {
                "seed": seed,
                "source": task.source,
                "task_id": task.task_id,
            }
        ),
    )

    selected = []
    excluded = Counter()
    seen_families = set()
    seen_prompts = set()

    for task in ordered:
        if task.source_split != "train":
            excluded["official_nontraining_split"] += 1
            continue

        try:
            assert_collectible(task)
        except ValueError:
            excluded["verifier_not_ready"] += 1
            continue

        if overlaps_legacy(
            task.raw_prompt,
            legacy_prompts,
        ):
            excluded["dataset_v1_overlap"] += 1
            continue

        family = (task.source, task.family_id)
        fingerprint = content_hash(task.raw_prompt)

        if family in seen_families or fingerprint in seen_prompts:
            excluded["duplicate_or_family_variant"] += 1
            continue

        seen_families.add(family)
        seen_prompts.add(fingerprint)

        if len(selected) >= max_tasks:
            excluded["sample_limit"] += 1
            continue

        selected.append(task)

    if not selected:
        raise ValueError("No eligible tasks after preflight")

    # Detect duplicate task identities even if prompts
    # differ: otherwise task lookup may be ambiguous.
    identities = [(task.source, task.task_id) for task in tasks]

    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate task identities")

    attempts = []

    for task in selected:
        prompt = render_execution_prompt(task)
        fingerprint = content_hash(task.raw_prompt)

        execution_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        for config in configurations:
            for repeat_index in range(repeats):
                identity = {
                    "protocol": PROTOCOL_VERSION,
                    "source": task.source,
                    "source_revision": (task.source_revision),
                    "task_id": task.task_id,
                    "raw_prompt_sha256": fingerprint,
                    "verifier": task.verifier,
                    "model_id": config["model_id"],
                    "effort": config["effort"],
                    "repeat_index": repeat_index,
                    "execution_prompt_sha256": execution_prompt_sha256,
                }

                attempts.append(
                    {
                        "attempt_id": stable_hash(identity),
                        "source": task.source,
                        "source_revision": (task.source_revision),
                        "task_id": task.task_id,
                        "family_id": task.family_id,
                        "domain": task.domain,
                        "raw_prompt_sha256": fingerprint,
                        "execution_prompt": prompt,
                        "verifier": task.verifier,
                        "model_id": config["model_id"],
                        "effort": config["effort"],
                        "repeat_index": repeat_index,
                        "max_output_tokens": (max_output_tokens),
                        "timeout_seconds": timeout_seconds,
                        "per_attempt_cap_usd": str(attempt_cap),
                        "execution_prompt_sha256": execution_prompt_sha256,
                    }
                )

    if len({attempt["attempt_id"] for attempt in attempts}) != len(attempts):
        raise ValueError("Duplicate planned attempt identities")

    plan = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "execution_enabled": False,
        "pool_split": "train",
        "seed": seed,
        "budget_usd": str(budget),
        "per_attempt_cap_usd": str(attempt_cap),
        "max_inflight": 1,
        "max_output_tokens": max_output_tokens,
        "timeout_seconds": timeout_seconds,
        "selected_tasks": len(selected),
        "planned_attempts": len(attempts),
        "candidates": configurations,
        "excluded": dict(sorted(excluded.items())),
        "attempts": attempts,
    }

    plan["plan_id"] = stable_hash(plan)

    return plan
