import json
from decimal import Decimal

import pytest

from backend.benchmarks.analytics import (
    analyze_benchmark,
    load_observations,
    percentile,
)


def make_records():
    """
    Two tasks, two reasoning efforts,
    two repetitions.

    Medium:
      Task A: correct
      Task B: incorrect

    High:
      Task A: correct
      Task B: correct
    """

    records = []

    for repeat_index in range(2):
        for task_id, domain in [
            ("task_a", "math"),
            ("task_b", "logic"),
        ]:
            for effort in ["medium", "high"]:
                success = task_id == "task_a" or effort == "high"

                records.append(
                    {
                        "run_id": 7,
                        "dataset_sha256": "a" * 64,
                        "task_id": task_id,
                        "domain": domain,
                        "model_id": "test-model",
                        "effort": effort,
                        "repeat_index": repeat_index,
                        "attempt_status": "completed",
                        "verification_status": ("correct" if success else "incorrect"),
                        "strict_success": success,
                        "cost_usd": ("0.10" if effort == "medium" else "0.20"),
                        "latency_ms": (100 if effort == "medium" else 200),
                    }
                )

    return records


def write_records(tmp_path, records):
    path = tmp_path / "benchmark.jsonl"

    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    return path


def make_report(tmp_path, records=None):
    records = records if records is not None else make_records()

    path = write_records(tmp_path, records)

    observations = load_observations(path)

    return analyze_benchmark(observations)


def test_configuration_metrics(tmp_path):
    report = make_report(tmp_path)

    configs = {config.effort: config for config in report.configurations}

    medium = configs["medium"]
    high = configs["high"]

    assert medium.terminal == 4
    assert high.terminal == 4

    assert medium.correct == 2
    assert high.correct == 4

    assert medium.strict_success_rate == 0.5
    assert high.strict_success_rate == 1.0

    assert medium.verified_accuracy == 0.5
    assert high.verified_accuracy == 1.0


def test_cost_metrics(tmp_path):
    report = make_report(tmp_path)

    configs = {config.effort: config for config in report.configurations}

    medium = configs["medium"]
    high = configs["high"]

    assert medium.total_known_cost_usd == Decimal("0.40")
    assert high.total_known_cost_usd == Decimal("0.80")

    assert medium.mean_known_cost_usd == Decimal("0.10")
    assert high.mean_known_cost_usd == Decimal("0.20")

    # Medium: $0.40 spent / 2 successes.
    assert medium.cost_per_success_usd == Decimal("0.20")

    # High: $0.80 spent / 4 successes.
    assert high.cost_per_success_usd == Decimal("0.20")


def test_latency_metrics(tmp_path):
    report = make_report(tmp_path)

    configs = {config.effort: config for config in report.configurations}

    assert configs["medium"].mean_latency_ms == 100
    assert configs["medium"].p95_latency_ms == 100

    assert configs["high"].mean_latency_ms == 200
    assert configs["high"].p90_latency_ms == 200

    assert (
        percentile(
            [100, 200, 300, 400],
            0.95,
        )
        == 400
    )


def test_effort_comparison(tmp_path):
    report = make_report(tmp_path)

    assert len(report.effort_comparisons) == 1

    comparison = report.effort_comparisons[0]

    assert comparison.lower_effort == "medium"
    assert comparison.higher_effort == "high"

    assert comparison.matched_attempts == 4

    assert comparison.improvements == 2
    assert comparison.regressions == 0

    assert comparison.success_rate_delta_pp == 50.0

    assert comparison.mean_cost_delta_usd == Decimal("0.10")


def test_pareto_frontier(tmp_path):
    report = make_report(tmp_path)

    # Medium is cheaper and faster.
    # High is more accurate.
    # Neither dominates the other.
    assert set(report.pareto_frontier) == {
        "test-model / medium",
        "test-model / high",
    }


def test_oracle(tmp_path):
    report = make_report(tmp_path)

    oracle = report.oracle

    assert oracle is not None

    assert oracle.comparable_cases == 4
    assert oracle.solved_cases == 4

    assert oracle.oracle_success_rate == 1.0

    # Task A, repeated twice: Medium at $0.10 each.
    # Task B, repeated twice: High at $0.20 each.
    #
    # Total = 2 * $0.10 + 2 * $0.20 = $0.60

    assert oracle.cheapest_successful_cost_usd == (Decimal("0.60"))

    assert oracle.selected_counts == {
        "test-model / medium": 2,
        "test-model / high": 2,
    }

    assert oracle.best_fixed_configuration == ("test-model / high")

    assert oracle.best_fixed_success_rate == 1.0


def test_domain_metrics(tmp_path):
    report = make_report(tmp_path)

    domain_results = {(item.domain, item.effort): item for item in report.domains}

    assert domain_results[("math", "medium")].strict_success_rate == 1.0

    assert domain_results[("logic", "medium")].strict_success_rate == 0.0

    assert domain_results[("logic", "high")].strict_success_rate == 1.0


def test_incomplete_experiment(tmp_path):
    records = make_records()

    records[0]["attempt_status"] = "pending"
    records[0]["strict_success"] = None
    records[0]["verification_status"] = None
    records[0]["cost_usd"] = None
    records[0]["latency_ms"] = None

    report = make_report(tmp_path, records)

    assert report.oracle is None
    assert report.pareto_frontier == []

    assert report.warnings


def test_missing_cost_suppresses_oracle(tmp_path):
    records = make_records()

    records[0]["cost_usd"] = None

    report = make_report(tmp_path, records)

    assert report.oracle is None
    assert report.pareto_frontier == []

    configs = {config.effort: config for config in report.configurations}

    assert configs["medium"].cost_per_success_usd is None


def test_duplicate_observation_rejected(tmp_path):
    records = make_records()

    records.append(records[0].copy())

    path = write_records(tmp_path, records)

    with pytest.raises(
        ValueError,
        match="Duplicate benchmark observation",
    ):
        load_observations(path)


def test_mixed_runs_rejected(tmp_path):
    records = make_records()

    records[0]["run_id"] = 99

    path = write_records(tmp_path, records)

    with pytest.raises(
        ValueError,
        match="multiple benchmark runs",
    ):
        load_observations(path)


def test_invalid_success_label(tmp_path):
    records = make_records()

    # Force an inconsistent success label.
    records[0]["strict_success"] = True
    records[0]["verification_status"] = "incorrect"

    path = write_records(tmp_path, records)

    with pytest.raises(
        ValueError,
        match="Inconsistent success label",
    ):
        load_observations(path)
