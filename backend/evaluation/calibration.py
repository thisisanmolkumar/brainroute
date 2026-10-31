import math

import numpy as np
from pydantic import BaseModel, Field

from backend.benchmarks.ml_dataset import MLExample
from backend.ml.metrics import evaluate_probabilities

EPS = 1e-7


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class TemperatureCalibrator(BaseModel):
    schema_version: int = 1
    temperature: float = Field(gt=0)

    def transform(self, probability: float) -> float:
        if not math.isfinite(probability):
            raise ValueError("Nonfinite probability")

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be in [0, 1]")

        p = min(max(probability, EPS), 1 - EPS)

        logit = math.log(p) - math.log1p(-p)

        return stable_sigmoid(logit / self.temperature)

    def transform_many(
        self,
        probabilities: list[float],
    ) -> list[float]:
        return [self.transform(value) for value in probabilities]


def fit_temperature(
    examples: list[MLExample],
    probabilities: list[float],
    *,
    min_groups: int = 20,
    min_outcomes_per_class: int = 10,
) -> tuple[TemperatureCalibrator, dict]:

    if len(examples) != len(probabilities):
        raise ValueError("Calibration input length mismatch")

    if not examples:
        raise ValueError("Empty calibration dataset")

    groups = {example.metadata.group_key for example in examples}

    successes = sum(example.targets.successes for example in examples)

    trials = sum(example.targets.trials for example in examples)

    failures = trials - successes

    if len(groups) < min_groups:
        raise ValueError(f"Need at least {min_groups} independent calibration groups")

    if min(successes, failures) < min_outcomes_per_class:
        raise ValueError(
            "Insufficient successes or failures for probability calibration"
        )

    # Validate probability inputs before optimization.
    before = evaluate_probabilities(
        examples,
        probabilities,
    )

    # A simple deterministic one-parameter search.
    # No additional optimization library required.
    temperatures = np.exp(
        np.linspace(
            math.log(0.25),
            math.log(10.0),
            401,
        )
    )

    def loss_for_temperature(
        temperature: float,
    ) -> float:
        calibrator = TemperatureCalibrator(temperature=temperature)

        predictions = calibrator.transform_many(probabilities)

        return evaluate_probabilities(
            examples,
            predictions,
        )["log_loss"]

    best_temperature = min(
        temperatures,
        key=loss_for_temperature,
    )

    calibrator = TemperatureCalibrator(temperature=float(best_temperature))

    after = evaluate_probabilities(
        examples,
        calibrator.transform_many(probabilities),
    )

    report = {
        "groups": len(groups),
        "trials": trials,
        "successes": successes,
        "failures": failures,
        "temperature": calibrator.temperature,
        "log_loss_before": before["log_loss"],
        "log_loss_after": after["log_loss"],
    }

    return calibrator, report
