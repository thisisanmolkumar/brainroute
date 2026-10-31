import argparse
import json
from decimal import Decimal
from pathlib import Path

from backend.core.model_registry import ModelRegistry
from backend.evaluation.runner import evaluate_router
from backend.ml.neural_training import NeuralRouterPredictor
from backend.routing.schemas import (
    CandidateConfig,
    RoutingPolicy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BrainRoute routing policies")

    parser.add_argument(
        "--mode",
        choices=["development", "heldout"],
        default="development",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/brainroute_neural_v1"),
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("datasets/ml/v1/train.jsonl"),
    )

    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("datasets/ml/v1/validation.jsonl"),
    )

    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("datasets/ml/v1/validation.jsonl"),
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--efforts",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--objective",
        choices=[
            "cost",
            "latency",
            "quality",
            "balanced",
        ],
        default="quality",
    )

    parser.add_argument(
        "--min-success",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-cost-usd",
        type=Decimal,
        default=None,
    )

    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--cost-reference-usd",
        type=Decimal,
        default=None,
    )

    parser.add_argument(
        "--latency-reference-ms",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation/router_evaluation.json"),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")

    configurations = [
        CandidateConfig(
            model_id=model,
            effort=effort,
        )
        for model in args.models
        for effort in args.efforts
    ]

    # Validate model/effort support with BrainRoute's
    # existing registry before evaluating them.
    registry = ModelRegistry()

    for config in configurations:
        entry = registry.validate_configuration(
            config.model_id,
            config.effort,
        )

        if not entry.enabled:
            raise ValueError(f"Disabled model: {config.model_id}")

    policy = RoutingPolicy(
        objective=args.objective,
        min_success_probability=args.min_success,
        max_cost_usd=args.max_cost_usd,
        max_latency_ms=args.max_latency_ms,
        cost_reference_usd=args.cost_reference_usd,
        latency_reference_ms=args.latency_reference_ms,
    )

    print("[3.5] Loading neural router...", flush=True)

    predictor = NeuralRouterPredictor(args.model_dir)

    print("[3.5] Evaluating predictions...", flush=True)

    report = evaluate_router(
        predictor=predictor,
        model_report_path=(args.model_dir / "report.json"),
        train_path=args.train,
        validation_path=args.validation,
        evaluation_path=args.evaluation,
        calibration_path=args.calibration,
        candidates=configurations,
        policy=policy,
        mode=args.mode,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n========== BRAINROUTE EVALUATION ==========")
    print("Mode:", report["evaluation_mode"])
    print("Tasks:", report["dataset"]["tasks"])
    print("Trials:", report["dataset"]["trials"])

    metrics = report["prediction_metrics"]

    print("\nSUCCESS PREDICTION")
    print(
        "Log loss:",
        metrics["success"]["log_loss"],
    )
    print(
        "Brier score:",
        metrics["success"]["brier_score"],
    )
    print(
        "ECE:",
        metrics["success"]["expected_calibration_error"],
    )

    print("\nPAIRWISE RANKING")
    print(
        "Accuracy:",
        metrics["pairwise_ranking"]["ranking_accuracy"],
    )

    replay = report["policy_replay"]

    print("\nPOLICY COVERAGE")
    print(
        "Complete-matrix tasks:",
        replay["task_coverage"]["complete_matrix_tasks"],
    )
    print(
        "Routed tasks:",
        replay["task_coverage"]["routed_tasks"],
    )
    print(
        "Abstentions:",
        replay["task_coverage"]["router_abstentions"],
    )

    print("\nPOLICY COMPARISON")

    for name, result in replay["policies"].items():
        if result is None:
            print(name, "No evaluated tasks")
            continue

        print(
            f"{name}: "
            f"success={result['mean_empirical_success']:.4f}, "
            f"cost={result['mean_cost_usd']}, "
            f"latency={result['mean_latency_ms']}"
        )

    for warning in report["warnings"]:
        print("WARNING:", warning)

    print("\nReport saved to:", args.output)


if __name__ == "__main__":
    main()
