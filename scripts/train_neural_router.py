import argparse
import json
from pathlib import Path

from backend.ml.neural_training import train_neural_router


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BrainRoute PyTorch neural router"
    )

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--embedding-config",
        type=Path,
        default=Path("models/brainroute_semantic_v1/embedding_config.json"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/brainroute_neural_v1"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--semantic-report",
        type=Path,
        default=None,
        help="Optional Phase 3.2 report for comparison",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    report = train_neural_router(
        train_path=args.train,
        validation_path=args.validation,
        embedding_config_path=args.embedding_config,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("\n========== NEURAL ROUTER RESULTS ==========")

    print(
        "Train groups:",
        report["dataset"]["train_groups"],
    )

    print(
        "Validation groups:",
        report["dataset"]["validation_groups"],
    )

    print(
        "Best epoch:",
        report["training"]["best_epoch"],
    )

    success = report["metrics"]["success"]
    cost = report["metrics"]["cost"]
    latency = report["metrics"]["latency"]

    print("\nSUCCESS PREDICTION")
    print(f"Log loss: {success['log_loss']:.6f}")
    print(f"Brier:    {success['brier_score']:.6f}")

    print("\nCOST PREDICTION")
    print(
        "Weighted MAE (USD):",
        cost["weighted_mae"],
    )
    print(
        "Evaluated trials:",
        cost["observations"],
    )

    print("\nLATENCY PREDICTION")
    print(
        "Weighted MAE (ms):",
        latency["weighted_mae"],
    )
    print(
        "Evaluated trials:",
        latency["observations"],
    )

    if args.semantic_report is not None:
        previous = json.loads(args.semantic_report.read_text(encoding="utf-8"))

        current_dataset = report["dataset"]
        previous_dataset = previous["dataset"]

        comparable = (
            current_dataset["train_sha256"] == previous_dataset["train_sha256"]
            and current_dataset["validation_sha256"]
            == previous_dataset["validation_sha256"]
        )

        if comparable:
            previous_metrics = previous["metrics"]["semantic_xgboost"]

            print("\nPHASE 3.2 COMPARISON")
            print(
                "Semantic XGBoost log loss:",
                f"{previous_metrics['log_loss']:.6f}",
            )
            print(
                "Neural router log loss:",
                f"{success['log_loss']:.6f}",
            )
        else:
            print(
                "\nWARNING: Phase 3.2 dataset hashes do not match. Comparison skipped."
            )

    for warning in report["warnings"]:
        print("WARNING:", warning)

    print("\nSaved model to:", args.output_dir)


if __name__ == "__main__":
    main()
