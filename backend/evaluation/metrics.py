from collections import defaultdict
from itertools import combinations

import numpy as np

from backend.benchmarks.ml_dataset import MLExample
from backend.ml.metrics import evaluate_probabilities


def probability_report(
    examples: list[MLExample],
    probabilities: list[float],
    bins: int = 10,
) -> dict:

    if bins < 2:
        raise ValueError("bins must be >= 2")

    summary = evaluate_probabilities(
        examples,
        probabilities,
    )

    bucket_data = [
        {
            "trials": 0,
            "successes": 0,
            "predicted_probability_sum": 0.0,
        }
        for _ in range(bins)
    ]

    for example, probability in zip(
        examples,
        probabilities,
        strict=True,
    ):
        index = min(
            int(float(probability) * bins),
            bins - 1,
        )

        trials = example.targets.trials

        bucket = bucket_data[index]

        bucket["trials"] += trials
        bucket["successes"] += example.targets.successes

        bucket["predicted_probability_sum"] += float(probability) * trials

    reliability = []
    weighted_error = 0.0
    total_trials = summary["trials"]

    for index, bucket in enumerate(bucket_data):
        trials = bucket["trials"]

        if trials == 0:
            continue

        predicted = bucket["predicted_probability_sum"] / trials

        observed = bucket["successes"] / trials

        weighted_error += (trials / total_trials) * abs(predicted - observed)

        reliability.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "trials": trials,
                "mean_predicted": predicted,
                "observed_success_rate": observed,
            }
        )

    summary["expected_calibration_error"] = weighted_error

    summary["reliability_bins"] = reliability

    summary["failures"] = total_trials - sum(
        example.targets.successes for example in examples
    )

    return summary


def pairwise_ranking_report(
    examples: list[MLExample],
    probabilities: list[float],
) -> dict:

    if len(examples) != len(probabilities):
        raise ValueError("Prediction count mismatch")

    grouped = defaultdict(list)

    for example, probability in zip(
        examples,
        probabilities,
        strict=True,
    ):
        key = (
            example.metadata.source,
            example.metadata.task_id,
        )

        grouped[key].append(
            (
                example.targets.success_rate,
                float(probability),
            )
        )

    comparable = 0
    concordant = 0.0
    tied_outcomes = 0

    for rows in grouped.values():
        for first, second in combinations(rows, 2):
            observed_delta = first[0] - second[0]
            predicted_delta = first[1] - second[1]

            if abs(observed_delta) < 1e-12:
                tied_outcomes += 1
                continue

            comparable += 1

            if abs(predicted_delta) < 1e-12:
                concordant += 0.5

            elif observed_delta * predicted_delta > 0:
                concordant += 1.0

    return {
        "comparable_pairs": comparable,
        "tied_outcome_pairs": tied_outcomes,
        "ranking_accuracy": (concordant / comparable if comparable else None),
    }


def regression_report(
    examples: list[MLExample],
    predictions: list[dict],
    *,
    target_field: str,
    prediction_field: str,
    mask_field: str,
) -> dict:

    if len(examples) != len(predictions):
        raise ValueError("Prediction count mismatch")

    errors = []
    squared_errors = []
    weights = []

    for example, prediction in zip(
        examples,
        predictions,
        strict=True,
    ):
        if not getattr(example.masks, mask_field):
            continue

        value = prediction.get(prediction_field)

        if value is None:
            continue

        actual = float(getattr(example.targets, target_field))

        difference = float(value) - actual

        errors.append(abs(difference))
        squared_errors.append(difference**2)
        weights.append(example.targets.trials)

    if not errors:
        return {
            "evaluated_examples": 0,
            "evaluated_trials": 0,
            "weighted_mae": None,
            "weighted_rmse": None,
        }

    return {
        "evaluated_examples": len(errors),
        "evaluated_trials": sum(weights),
        "weighted_mae": float(np.average(errors, weights=weights)),
        "weighted_rmse": float(
            np.sqrt(
                np.average(
                    squared_errors,
                    weights=weights,
                )
            )
        ),
    }
