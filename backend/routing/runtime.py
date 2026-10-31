import os
from functools import lru_cache
from pathlib import Path

from backend.core.model_registry import ModelRegistry
from backend.ml.neural_training import NeuralRouterPredictor
from backend.routing.service import RoutingService


@lru_cache(maxsize=1)
def get_routing_service() -> RoutingService:
    model_dir = Path(
        os.getenv(
            "BRAINROUTE_ROUTER_MODEL_DIR",
            "models/brainroute_neural_v1",
        )
    )

    train_path = Path(
        os.getenv(
            "BRAINROUTE_ROUTER_TRAIN_DATA",
            "datasets/ml/v1/train.jsonl",
        )
    )

    validation_path = Path(
        os.getenv(
            "BRAINROUTE_ROUTER_VALIDATION_DATA",
            "datasets/ml/v1/validation.jsonl",
        )
    )

    predictor = NeuralRouterPredictor(
        model_dir=model_dir,
    )

    registry = ModelRegistry()

    return RoutingService(
        predictor=predictor,
        registry=registry,
        train_path=train_path,
        validation_path=validation_path,
        report_path=model_dir / "report.json",
    )
