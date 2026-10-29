import argparse
from decimal import Decimal
from pathlib import Path

from backend.core.model_registry import ModelRegistry
from backend.ml.neural_training import NeuralRouterPredictor
from backend.routing.schemas import (
    CandidateConfig,
    RoutingPolicy,
)
from backend.routing.service import RoutingService


def parse_args():
    parser = argparse.ArgumentParser(description="BrainRoute Intelligent Routing")

    prompt_group = parser.add_mutually_exclusive_group(required=True)

    prompt_group.add_argument("--prompt")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
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
        default="cost",
    )

    parser.add_argument(
        "--min-success",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--max-cost-usd",
        type=Decimal,
    )

    parser.add_argument(
        "--max-latency-ms",
        type=float,
    )

    parser.add_argument(
        "--cost-reference-usd",
        type=Decimal,
    )

    parser.add_argument(
        "--latency-reference-ms",
        type=float,
    )

    parser.add_argument(
        "--success-weight",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--latency-weight",
        type=float,
        default=0.15,
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
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def format_cost(value):
    if value is None:
        return "N/A"

    return f"${value:.8f}"


def format_latency(value):
    if value is None:
        return "N/A"

    return f"{value:.0f} ms"


def main():
    args = parse_args()

    prompt = (
        args.prompt
        if args.prompt is not None
        else args.prompt_file.read_text(encoding="utf-8")
    )

    configs = [
        CandidateConfig(
            model_id=model,
            effort=effort,
        )
        for model in args.models
        for effort in args.efforts
    ]

    policy = RoutingPolicy(
        objective=args.objective,
        min_success_probability=args.min_success,
        max_cost_usd=args.max_cost_usd,
        max_latency_ms=args.max_latency_ms,
        cost_reference_usd=args.cost_reference_usd,
        latency_reference_ms=args.latency_reference_ms,
        success_weight=args.success_weight,
        cost_weight=args.cost_weight,
        latency_weight=args.latency_weight,
    )

    print(
        "[3.4] Loading NeuralRouterPredictor...",
        flush=True,
    )

    predictor = NeuralRouterPredictor(args.model_dir)

    service = RoutingService(
        predictor=predictor,
        registry=ModelRegistry(),
        train_path=args.train,
        validation_path=args.validation,
        report_path=args.model_dir / "report.json",
    )

    decision = service.route(
        prompt=prompt,
        configs=configs,
        policy=policy,
    )

    print("\n========== BRAINROUTE ROUTING ==========")

    print("Objective:", decision.objective)
    print("Status:", decision.status)
    print("Automatic execution: DISABLED")

    print("\nCANDIDATE ASSESSMENTS")

    ordered = sorted(
        decision.assessments,
        key=lambda item: (
            item.rank is None,
            item.rank if item.rank is not None else 999,
            item.estimate.model_id,
            item.estimate.effort,
        ),
    )

    for item in ordered:
        estimate = item.estimate

        name = f"{estimate.model_id} / {estimate.effort}"

        print(f"\n{name}")
        print(
            "  Predicted success:",
            f"{estimate.success_probability:.2%}",
        )
        print(
            "  Predicted cost:",
            format_cost(estimate.predicted_cost_usd),
        )
        print(
            "  Predicted latency:",
            format_latency(estimate.predicted_latency_ms),
        )

        if item.eligible:
            print("  Rank:", item.rank)
            print("  Pareto-dominated:", item.dominated)

        else:
            print("  Excluded:")

            for reason in item.rejection_reasons:
                print("   -", reason)

    print("\n========== RECOMMENDATION ==========")

    if decision.recommendation is None:
        print("No candidate satisfies the predicted constraints.")

    else:
        candidate = decision.recommendation

        print(
            candidate.model_id,
            "/",
            candidate.effort,
        )

        print(
            "Predicted success:",
            f"{candidate.success_probability:.2%}",
        )

        print(
            "Predicted cost:",
            format_cost(candidate.predicted_cost_usd),
        )

        print(
            "Predicted latency:",
            format_latency(candidate.predicted_latency_ms),
        )

    if decision.pareto_frontier:
        print("\nPARETO FRONTIER")

        for name in decision.pareto_frontier:
            print("-", name)

    if decision.readiness_blockers:
        print("\nREVIEW REQUIRED")

        for reason in decision.readiness_blockers:
            print("-", reason)

    if decision.warnings:
        print("\nWARNINGS")

        for warning in decision.warnings:
            print("-", warning)

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            decision.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        print("\nDecision saved to:", args.output)


if __name__ == "__main__":
    main()
