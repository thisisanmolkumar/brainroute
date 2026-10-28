import math
from collections import defaultdict
from collections.abc import Sequence

from backend.benchmarks.ml_dataset import MLExample

EPSILON = 1e-7


def evaluate_probabilities(
    examples: list[MLExample],
    probabilities: Sequence[float],
) -> dict:

    if not examples:
        raise ValueError("Cannot evaluate empty dataset")

    if len(examples) != len(probabilities):
        raise ValueError("Prediction count does not match example count")

    total_trials = 0
    total_successes = 0

    logloss_sum = 0.0
    brier_sum = 0.0

    for example, probability in zip(
        examples,
        probabilities,
        strict=True,
    ):
        p = float(probability)

        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"Invalid success probability: {p}")

        p = min(
            max(p, EPSILON),
            1 - EPSILON,
        )

        successes = example.targets.successes
        trials = example.targets.trials
        failures = trials - successes

        total_trials += trials
        total_successes += successes

        # Aggregated binomial log loss
        logloss_sum += -(successes * math.log(p) + failures * math.log(1 - p))

        # Attempt-level Brier score
        brier_sum += successes * (1 - p) ** 2 + failures * p**2

    return {
        "examples": len(examples),
        "trials": total_trials,
        "observed_success_rate": (total_successes / total_trials),
        "log_loss": logloss_sum / total_trials,
        "brier_score": brier_sum / total_trials,
    }


def global_prior_predictions(
    train: list[MLExample],
    validation: list[MLExample],
) -> list[float]:
    """
    Predict the same training-derived success
    probability for every validation example.

    Laplace smoothing avoids exact 0 and 1.
    """

    successes = sum(item.targets.successes for item in train)

    trials = sum(item.targets.trials for item in train)

    probability = (successes + 1) / (trials + 2)

    return [probability] * len(validation)


def configuration_prior_predictions(
    train: list[MLExample],
    validation: list[MLExample],
) -> list[float]:
    """
    Predict historical success probability
    for each model/effort configuration.

    This is our stronger non-ML baseline.
    """

    statistics = defaultdict(lambda: [0, 0])

    global_successes = 0
    global_trials = 0

    for item in train:
        key = (
            item.features.model_id,
            item.features.effort,
        )

        successes = item.targets.successes
        trials = item.targets.trials

        statistics[key][0] += successes
        statistics[key][1] += trials

        global_successes += successes
        global_trials += trials

    global_probability = (global_successes + 1) / (global_trials + 2)

    predictions = []

    for item in validation:
        key = (
            item.features.model_id,
            item.features.effort,
        )

        if key not in statistics:
            predictions.append(global_probability)
            continue

        successes, trials = statistics[key]

        predictions.append((successes + 1) / (trials + 2))

    return predictions
