import hashlib
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from backend.datasets.v2_schema import TaskV2

GSM8K_FINAL_RE = re.compile(
    r"####\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?)\s*$"
)


def require_string(
    record: Mapping[str, Any],
    field: str,
) -> str:
    value = record.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid string field: {field}")

    return value


def require_list_of_strings(
    record: Mapping[str, Any],
    field: str,
) -> list[str]:
    value = record.get(field)

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected a list of strings: {field}")

    return value


def gsm8k_final_answer(answer: str) -> str:
    match = GSM8K_FINAL_RE.search(answer)

    if match is None:
        raise ValueError("GSM8K solution is missing its final '#### number' answer")

    raw = match.group(1).replace(",", "")

    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError("Invalid GSM8K numeric reference") from exc

    if not value.is_finite():
        raise ValueError("Nonfinite GSM8K reference answer")

    return format(value, "f")


def stable_prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]


def adapt_gsm8k(
    record: Mapping[str, Any],
    *,
    revision: str,
    split: str,
    index: int,
) -> TaskV2:

    if split not in {"train", "test"}:
        raise ValueError("GSM8K main supports train/test splits")

    prompt = require_string(record, "question")
    answer = require_string(record, "answer")

    reference = gsm8k_final_answer(answer)

    return TaskV2(
        task_id=(f"main/{split}/{stable_prompt_id(prompt)}"),
        source="gsm8k",
        source_revision=revision,
        source_split=split,
        domain="math",
        difficulty="unknown",
        raw_prompt=prompt,
        reference_answer=reference,
        verifier="gsm8k_final_number",
        family_id=(f"gsm8k/{stable_prompt_id(prompt)}"),
        verification_status="ready",
        metadata={
            "upstream_dataset": "openai/gsm8k",
            "upstream_config": "main",
            "upstream_split": split,
            "row_index": index,
            "answer_format": "final_number",
        },
    )


def adapt_mbpp(
    record: Mapping[str, Any],
    *,
    revision: str,
    split: str,
    config: str,
) -> TaskV2:

    if config not in {"sanitized", "full"}:
        raise ValueError("Unsupported MBPP configuration")

    if split == "prompt":
        raise ValueError(
            "MBPP few-shot prompt partition must "
            "not be imported for training/evaluation"
        )

    if split not in {
        "train",
        "validation",
        "evaluation",
        "test",
    }:
        raise ValueError("Unknown MBPP split")

    source_split = "validation" if split == "evaluation" else split

    task_id = record.get("task_id")

    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise TypeError("MBPP requires an integer task_id")

    prompt_field = "prompt" if config == "sanitized" else "text"

    prompt = require_string(record, prompt_field)

    tests = require_list_of_strings(
        record,
        "test_list",
    )

    if not tests:
        raise ValueError("MBPP task has no verification tests")

    if config == "sanitized":
        setup = require_list_of_strings(
            record,
            "test_imports",
        )
    else:
        setup_code = record.get(
            "test_setup_code",
            "",
        )

        if not isinstance(setup_code, str):
            raise ValueError("Invalid MBPP test_setup_code")

        setup = [setup_code] if setup_code else []

    reference_solution = record.get("code", "")

    if not isinstance(reference_solution, str):
        raise TypeError("Invalid reference solution")

    solution_hash = hashlib.sha256(reference_solution.encode("utf-8")).hexdigest()

    return TaskV2(
        task_id=f"{config}/{task_id}",
        source="mbpp",
        source_revision=revision,
        source_split=source_split,
        domain="coding",
        difficulty="unknown",
        raw_prompt=prompt,
        reference_answer=None,
        verifier=None,
        family_id=f"mbpp/{task_id}",
        verification_status="pending",
        metadata={
            "upstream_dataset": ("google-research-datasets/mbpp"),
            "upstream_config": config,
            "upstream_split": split,
            "mbpp_task_id": task_id,
            "test_list": tests,
            "test_setup": setup,
            "reference_solution_sha256": solution_hash,
            "requires_isolated_execution": True,
        },
    )


def adapt_bbh(
    record: Mapping[str, Any],
    *,
    revision: str,
    split: str,
    config: str,
    index: int,
) -> TaskV2:

    if split != "test":
        raise ValueError("BBH examples must remain in the official test partition")

    if not config:
        raise ValueError("BBH task configuration required")

    prompt = require_string(record, "input")
    target = require_string(record, "target")

    # Initially, only the Boolean-expression subset
    # has an implemented, explicit verifier.
    boolean_ready = config == "boolean_expressions" and target in {"True", "False"}

    return TaskV2(
        task_id=f"{config}/{index:05d}",
        source="bbh",
        source_revision=revision,
        source_split="test",
        domain="logic",
        difficulty="unknown",
        raw_prompt=prompt,
        reference_answer=target,
        verifier=("bbh_boolean" if boolean_ready else None),
        family_id=f"bbh/{config}/{index:05d}",
        verification_status=("ready" if boolean_ready else "pending"),
        metadata={
            "upstream_dataset": "lukaemon/bbh",
            "upstream_config": config,
            "upstream_split": split,
            "row_index": index,
        },
    )


def adapt_ifeval(
    record: Mapping[str, Any],
    *,
    revision: str,
    split: str,
) -> TaskV2:

    # Hugging Face calls this split "train", but
    # BrainRoute treats these published benchmark
    # prompts as evaluation-only.
    if split != "train":
        raise ValueError("Unexpected IFEval source split")

    key = record.get("key")

    if isinstance(key, bool) or not isinstance(key, int):
        raise TypeError("IFEval requires an integer key")

    prompt = require_string(record, "prompt")

    instruction_ids = require_list_of_strings(
        record,
        "instruction_id_list",
    )

    kwargs = record.get("kwargs")

    if (
        not instruction_ids
        or not isinstance(kwargs, list)
        or len(kwargs) != len(instruction_ids)
        or not all(isinstance(item, dict) for item in kwargs)
    ):
        raise ValueError("Invalid IFEval instruction metadata")

    return TaskV2(
        task_id=f"ifeval/{key}",
        source="ifeval",
        source_revision=revision,
        source_split="test",
        domain="instruction",
        difficulty="unknown",
        raw_prompt=prompt,
        reference_answer=None,
        verifier=None,
        family_id=f"ifeval/{key}",
        verification_status="pending",
        metadata={
            "upstream_dataset": "google/IFEval",
            "upstream_split": split,
            "instruction_ids": instruction_ids,
            "instruction_kwargs": kwargs,
            "official_verifier_required": True,
        },
    )


def adapt_record(
    source: str,
    record: Mapping[str, Any],
    *,
    revision: str,
    split: str,
    index: int,
    config: str | None = None,
) -> TaskV2:

    if not revision.strip():
        raise ValueError("Source revision is required")

    if source == "gsm8k":
        if config not in {None, "main"}:
            raise ValueError("Unsupported GSM8K config")

        return adapt_gsm8k(
            record,
            revision=revision,
            split=split,
            index=index,
        )

    if source == "mbpp":
        return adapt_mbpp(
            record,
            revision=revision,
            split=split,
            config=config or "sanitized",
        )

    if source == "bbh":
        return adapt_bbh(
            record,
            revision=revision,
            split=split,
            config=config or "",
            index=index,
        )

    if source == "ifeval":
        if config is not None:
            raise ValueError("IFEval does not use a task config")

        return adapt_ifeval(
            record,
            revision=revision,
            split=split,
        )

    raise ValueError(f"Unsupported dataset: {source}")
