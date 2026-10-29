import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from backend.core.model_registry import ModelRegistry
from backend.ml.data import (
    group_count,
    load_ml_examples,
    validate_training_splits,
)
from backend.routing.optimizer import optimize_route
from backend.routing.schemas import (
    CandidateConfig,
    CandidateEstimate,
    RoutingDecision,
    RoutingPolicy,
)


class RoutingPredictor(Protocol):
    preprocessing: dict

    def predict_many(
        self,
        requests: list[tuple[str, str, str]],
    ) -> list[dict]: ...


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RoutingService:
    def __init__(
        self,
        predictor: RoutingPredictor,
        registry: ModelRegistry,
        train_path: str | Path,
        validation_path: str | Path,
        report_path: str | Path,
        *,
        min_validation_groups: int = 20,
        min_train_groups_per_config: int = 20,
        min_outcomes_per_class: int = 5,
    ) -> None:

        self.predictor = predictor
        self.registry = registry

        train_path = Path(train_path)
        validation_path = Path(validation_path)
        report_path = Path(report_path)

        self.train = load_ml_examples(train_path)
        self.validation = load_ml_examples(validation_path)

        validate_training_splits(
            self.train,
            self.validation,
        )

        self.report = json.loads(report_path.read_text(encoding="utf-8"))

        if self.report.get("model_type") != ("pytorch_multitask_router"):
            raise ValueError("Expected a Phase 3.3 neural-router report")

        dataset = self.report["dataset"]

        if dataset["train_sha256"] != file_sha256(train_path) or dataset[
            "validation_sha256"
        ] != file_sha256(validation_path):
            raise ValueError(
                "Training/validation dataset hashes do not "
                "match the saved neural-router report"
            )

        self.min_validation_groups = min_validation_groups

        self.min_train_groups_per_config = min_train_groups_per_config

        self.min_outcomes_per_class = min_outcomes_per_class

        self.training_by_config = defaultdict(list)

        for example in self.train:
            key = (
                example.features.model_id,
                example.features.effort,
            )

            self.training_by_config[key].append(example)

    def _validate_configs(
        self,
        configs: list[CandidateConfig],
    ) -> None:

        if not configs:
            raise ValueError("At least one candidate is required")

        if len(configs) > 24:
            raise ValueError("Maximum 24 routing candidates")

        identities = [(config.model_id, config.effort) for config in configs]

        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate routing candidates")

        supported_models = set(self.predictor.preprocessing["models"])

        supported_efforts = set(self.predictor.preprocessing["efforts"])

        for config in configs:
            model = self.registry.validate_configuration(
                config.model_id,
                config.effort,
            )

            if not model.enabled:
                raise ValueError(f"Model is disabled: {config.model_id}")

            if model.provider != "openai":
                raise ValueError("This version supports OpenAI configurations")

            if (
                config.model_id not in supported_models
                or config.effort not in supported_efforts
            ):
                raise ValueError(
                    f"Configuration unknown to trained router: "
                    f"{config.model_id}/{config.effort}"
                )

            # A model and an effort may each be known
            # separately, while their particular pair
            # was never evaluated during training.
            key = (config.model_id, config.effort)

            if not self.training_by_config.get(key):
                raise ValueError(
                    f"No training examples for {config.model_id}/{config.effort}"
                )

    def _readiness(
        self,
        configs: list[CandidateConfig],
    ) -> tuple[list[str], list[str]]:

        blockers = []
        warnings = []

        validation_groups = group_count(self.validation)

        if validation_groups < self.min_validation_groups:
            blockers.append(
                f"Only {validation_groups} distinct "
                "validation tasks; routing quality "
                "has not been established."
            )

        validation_successes = sum(item.targets.successes for item in self.validation)

        validation_trials = sum(item.targets.trials for item in self.validation)

        if validation_successes in {
            0,
            validation_trials,
        }:
            blockers.append(
                "Validation contains only one observed success/failure outcome class."
            )

        for config in configs:
            key = (config.model_id, config.effort)
            examples = self.training_by_config[key]

            name = f"{config.model_id}/{config.effort}"

            task_groups = group_count(examples)

            successes = sum(item.targets.successes for item in examples)

            trials = sum(item.targets.trials for item in examples)

            failures = trials - successes

            if task_groups < self.min_train_groups_per_config:
                blockers.append(f"{name}: only {task_groups} training task groups.")

            if (
                successes < self.min_outcomes_per_class
                or failures < self.min_outcomes_per_class
            ):
                blockers.append(
                    f"{name}: insufficient examples of "
                    "both successful and failed executions."
                )

            known_cost_examples = sum(item.masks.cost for item in examples)

            known_latency_examples = sum(item.masks.latency for item in examples)

            if known_cost_examples < len(examples):
                warnings.append(f"{name}: incomplete historical cost coverage.")

            if known_latency_examples < len(examples):
                warnings.append(f"{name}: incomplete historical latency coverage.")

        warnings.append(
            "Predicted cost is not a billing cap, and predicted latency is not an SLA."
        )

        warnings.append(
            "Predicted success probabilities are not yet independently calibrated."
        )

        return blockers, warnings

    def route(
        self,
        prompt: str,
        configs: list[CandidateConfig],
        policy: RoutingPolicy,
    ) -> RoutingDecision:

        if not prompt.strip():
            raise ValueError("Routing prompt cannot be empty")

        self._validate_configs(configs)

        # One batched neural inference call.
        requests = [
            (
                prompt,
                config.model_id,
                config.effort,
            )
            for config in configs
        ]

        raw_predictions = self.predictor.predict_many(requests)

        if len(raw_predictions) != len(configs):
            raise ValueError("Predictor returned an unexpected number of predictions")

        estimates = []

        for config, raw in zip(
            configs,
            raw_predictions,
            strict=True,
        ):
            if raw["model_id"] != config.model_id or raw["effort"] != config.effort:
                raise ValueError("Predictor configuration mismatch")

            key = (
                config.model_id,
                config.effort,
            )

            training_examples = self.training_by_config[key]

            # A regression head may have been trained
            # using other configurations but have no
            # measured labels for this one.
            if not any(item.masks.cost for item in training_examples):
                raw = {
                    **raw,
                    "predicted_cost_usd": None,
                }

            if not any(item.masks.latency for item in training_examples):
                raw = {
                    **raw,
                    "predicted_latency_ms": None,
                }

            estimates.append(CandidateEstimate.model_validate(raw))

        blockers, warnings = self._readiness(configs)

        return optimize_route(
            estimates,
            policy,
            readiness_blockers=blockers,
            warnings=warnings,
        )
