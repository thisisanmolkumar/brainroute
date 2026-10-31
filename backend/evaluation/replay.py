from collections import defaultdict

import numpy as np

from backend.benchmarks.ml_dataset import MLExample
from backend.routing.optimizer import optimize_route
from backend.routing.schemas import (
    CandidateConfig,
    CandidateEstimate,
    RoutingPolicy,
)


def identity(example: MLExample) -> tuple[str, str]:
    return (
        example.features.model_id,
        example.features.effort,
    )


def task_identity(
    example: MLExample,
) -> tuple[str, str]:
    return (
        example.metadata.source,
        example.metadata.task_id,
    )


def training_baselines(
    train: list[MLExample],
    candidates: list[CandidateConfig],
) -> dict:

    grouped = defaultdict(list)

    for example in train:
        grouped[identity(example)].append(example)

    config_keys = [(item.model_id, item.effort) for item in candidates]

    if not config_keys or len(config_keys) != len(set(config_keys)):
        raise ValueError("Candidate configurations must be unique")

    stats = {}

    for key in config_keys:
        examples = grouped.get(key, [])

        if not examples:
            raise ValueError(f"No training observations for {key}")

        successes = sum(item.targets.successes for item in examples)

        trials = sum(item.targets.trials for item in examples)

        known_cost = all(item.masks.cost for item in examples)

        cost = (
            sum(
                float(item.targets.mean_cost_usd) * item.targets.trials
                for item in examples
            )
            / trials
            if known_cost
            else None
        )

        stats[key] = {
            "success_rate": successes / trials,
            "mean_cost_usd": cost,
        }

    best = min(
        config_keys,
        key=lambda key: (
            -stats[key]["success_rate"],
            (
                stats[key]["mean_cost_usd"]
                if stats[key]["mean_cost_usd"] is not None
                else float("inf")
            ),
            key,
        ),
    )

    cost_candidates = [
        key for key in config_keys if stats[key]["mean_cost_usd"] is not None
    ]

    cheapest = (
        min(
            cost_candidates,
            key=lambda key: (
                stats[key]["mean_cost_usd"],
                key,
            ),
        )
        if cost_candidates
        else None
    )

    return {
        "fixed_best": best,
        "fixed_cheapest": cheapest,
        "training_statistics": {
            f"{key[0]}/{key[1]}": value for key, value in stats.items()
        },
    }


def observed_outcome(example: MLExample) -> dict:
    return {
        "success": example.targets.success_rate,
        "cost": (float(example.targets.mean_cost_usd) if example.masks.cost else None),
        "latency": (
            float(example.targets.mean_latency_ms) if example.masks.latency else None
        ),
    }


def average_outcomes(
    outcomes: list[dict],
) -> dict:
    """
    Exact expectation under a uniform random policy.

    Cost/latency are unknown if any candidate lacks
    the corresponding measured outcome.
    """
    return {
        "success": float(np.mean([item["success"] for item in outcomes])),
        "cost": (
            float(np.mean([item["cost"] for item in outcomes]))
            if all(item["cost"] is not None for item in outcomes)
            else None
        ),
        "latency": (
            float(np.mean([item["latency"] for item in outcomes]))
            if all(item["latency"] is not None for item in outcomes)
            else None
        ),
    }


def summarize_policy(
    records: list[dict],
    policy_name: str,
) -> dict | None:

    if not records:
        return None

    outcomes = [row["outcomes"][policy_name] for row in records]

    known_costs = [item["cost"] for item in outcomes if item["cost"] is not None]

    known_latencies = [
        item["latency"] for item in outcomes if item["latency"] is not None
    ]

    mean_success = float(np.mean([item["success"] for item in outcomes]))

    costs_complete = len(known_costs) == len(records)

    return {
        "tasks": len(records),
        "mean_empirical_success": mean_success,
        "mean_cost_usd": (float(np.mean(known_costs)) if known_costs else None),
        "cost_coverage_tasks": len(known_costs),
        "mean_latency_ms": (
            float(np.mean(known_latencies)) if known_latencies else None
        ),
        "latency_coverage_tasks": len(known_latencies),
        "cost_per_expected_success_usd": (
            sum(known_costs) / sum(item["success"] for item in outcomes)
            if costs_complete and sum(item["success"] for item in outcomes) > 0
            else None
        ),
    }


def paired_bootstrap_interval(
    records: list[dict],
    comparator: str,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict:

    grouped = defaultdict(list)

    for record in records:
        grouped[record["group_key"]].append(
            record["outcomes"]["neural"]["success"]
            - record["outcomes"][comparator]["success"]
        )

    group_keys = sorted(grouped)

    point_estimate = float(
        np.mean(
            [
                difference
                for differences in grouped.values()
                for difference in differences
            ]
        )
    )

    if len(group_keys) < 20:
        return {
            "difference": point_estimate,
            "ci_95": None,
            "reason": ("Fewer than 20 independent task groups"),
        }

    rng = np.random.default_rng(seed)

    estimates = []

    for _ in range(samples):
        selected = rng.integers(
            0,
            len(group_keys),
            size=len(group_keys),
        )

        differences = [
            difference
            for index in selected
            for difference in grouped[group_keys[int(index)]]
        ]

        estimates.append(float(np.mean(differences)))

    lower, upper = np.quantile(
        estimates,
        [0.025, 0.975],
    )

    return {
        "difference": point_estimate,
        "ci_95": [
            float(lower),
            float(upper),
        ],
        "reason": None,
    }


def replay_policies(
    train: list[MLExample],
    evaluation: list[MLExample],
    predictions: list[dict],
    candidates: list[CandidateConfig],
    policy: RoutingPolicy,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict:

    if len(evaluation) != len(predictions):
        raise ValueError("Prediction/evaluation length mismatch")

    baselines = training_baselines(
        train,
        candidates,
    )

    expected_keys = {(item.model_id, item.effort) for item in candidates}

    tasks = defaultdict(dict)
    predicted = defaultdict(dict)

    for example, prediction in zip(
        evaluation,
        predictions,
        strict=True,
    ):
        task = task_identity(example)
        config = identity(example)

        if config not in expected_keys:
            continue

        if config in tasks[task]:
            raise ValueError(f"Duplicate task/configuration: {task}/{config}")

        if (
            prediction["model_id"],
            prediction["effort"],
        ) != config:
            raise ValueError("Prediction order mismatch")

        tasks[task][config] = example
        predicted[task][config] = CandidateEstimate.model_validate(prediction)

    complete_tasks = 0
    excluded_incomplete = 0
    abstained = 0

    records = []

    for task in sorted(tasks):
        rows = tasks[task]

        if set(rows) != expected_keys:
            excluded_incomplete += 1
            continue

        complete_tasks += 1

        ordered_configs = [(item.model_id, item.effort) for item in candidates]

        estimates = [predicted[task][config] for config in ordered_configs]

        decision = optimize_route(
            estimates,
            policy,
        )

        if decision.recommendation is None:
            abstained += 1
            continue

        chosen_key = (
            decision.recommendation.model_id,
            decision.recommendation.effort,
        )

        observations = {
            config: observed_outcome(rows[config]) for config in ordered_configs
        }

        # The oracle is allowed to use observed
        # evaluation outcomes: an optimistic
        # hindsight benchmark, not a deployable policy.
        oracle_key = min(
            ordered_configs,
            key=lambda config: (
                -observations[config]["success"],
                (
                    observations[config]["cost"]
                    if observations[config]["cost"] is not None
                    else float("inf")
                ),
                config,
            ),
        )

        outcomes = {
            "neural": observations[chosen_key],
            "fixed_best": observations[baselines["fixed_best"]],
            "random_uniform": average_outcomes(list(observations.values())),
            "oracle": observations[oracle_key],
        }

        if baselines["fixed_cheapest"] is not None:
            outcomes["fixed_cheapest"] = observations[baselines["fixed_cheapest"]]

        selected_example = rows[chosen_key]

        records.append(
            {
                "task": f"{task[0]}/{task[1]}",
                "group_key": selected_example.metadata.group_key,
                "chosen": f"{chosen_key[0]}/{chosen_key[1]}",
                "outcomes": outcomes,
            }
        )

    policy_names = [
        "neural",
        "fixed_best",
        "random_uniform",
        "oracle",
    ]

    if baselines["fixed_cheapest"] is not None:
        policy_names.insert(2, "fixed_cheapest")

    summaries = {name: summarize_policy(records, name) for name in policy_names}

    comparisons = {}

    for name in policy_names:
        if name in {"neural", "oracle"} or not records:
            continue

        comparisons[name] = paired_bootstrap_interval(
            records,
            name,
            samples=bootstrap_samples,
            seed=seed,
        )

    # Historical, descriptive constraint checks.
    neural_outcomes = [row["outcomes"]["neural"] for row in records]

    success_below_threshold = sum(
        item["success"] < policy.min_success_probability for item in neural_outcomes
    )

    measured_cost = [item for item in neural_outcomes if item["cost"] is not None]

    measured_latency = [item for item in neural_outcomes if item["latency"] is not None]

    violations = {
        "empirical_success_below_threshold": (success_below_threshold),
        "evaluated_tasks": len(records),
        "actual_cost_over_limit": (
            sum(item["cost"] > float(policy.max_cost_usd) for item in measured_cost)
            if policy.max_cost_usd is not None
            else None
        ),
        "cost_measured_tasks": len(measured_cost),
        "actual_latency_over_limit": (
            sum(item["latency"] > policy.max_latency_ms for item in measured_latency)
            if policy.max_latency_ms is not None
            else None
        ),
        "latency_measured_tasks": len(measured_latency),
    }

    return {
        "task_coverage": {
            "tasks_with_any_candidate_data": len(tasks),
            "complete_matrix_tasks": complete_tasks,
            "excluded_incomplete_matrix": excluded_incomplete,
            "router_abstentions": abstained,
            "routed_tasks": len(records),
            "routing_coverage": (
                len(records) / complete_tasks if complete_tasks else None
            ),
        },
        "fixed_configurations": {
            "best_success": (
                f"{baselines['fixed_best'][0]}/{baselines['fixed_best'][1]}"
            ),
            "cheapest": (
                f"{baselines['fixed_cheapest'][0]}/{baselines['fixed_cheapest'][1]}"
                if baselines["fixed_cheapest"] is not None
                else None
            ),
        },
        "policies": summaries,
        "paired_success_differences": comparisons,
        "constraint_checks": violations,
        "warnings": [
            "All policy summaries use the same router-selected task cohort.",
            "Abstained and incomplete-matrix tasks are reported separately.",
            (
                "Oracle uses evaluation outcomes and is "
                "an optimistic retrospective benchmark."
            ),
            (
                "Task means are based on historical "
                "executions, not live router deployments."
            ),
        ],
    }
