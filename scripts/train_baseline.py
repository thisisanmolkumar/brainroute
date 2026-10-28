import argparse
from pathlib import Path

from backend.ml.baseline import train_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="Train BrainRoute XGBoost baseline")

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
        "--output-dir",
        type=Path,
        default=Path("models/brainroute_xgb_v1"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--allow-small-data",
        action="store_true",
        help="For pipeline testing, not reliable evaluation",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n========== BRAINROUTE TRAINING ==========")

    print("Training data:", args.train)
    print("Validation data:", args.validation)

    report = train_baseline(
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir,
        seed=args.seed,
        num_boost_round=args.rounds,
        early_stopping_rounds=(args.early_stopping_rounds),
        allow_small_data=args.allow_small_data,
    )

    print("\n========== TRAINING RESULTS ==========")

    dataset = report["dataset"]

    print("Training examples:", dataset["train_examples"])
    print("Validation examples:", dataset["validation_examples"])

    print("Training groups:", dataset["train_groups"])
    print("Validation groups:", dataset["validation_groups"])

    print(
        "Best iteration:",
        report["training"]["best_iteration"],
    )

    print("\n========== VALIDATION METRICS ==========")

    for name, metrics in report["metrics"].items():
        print(f"\n{name}")

        print(
            "Log loss:",
            f"{metrics['log_loss']:.6f}",
        )

        print(
            "Brier score:",
            f"{metrics['brier_score']:.6f}",
        )

    if report["warnings"]:
        print("\n========== WARNINGS ==========")

        for warning in report["warnings"]:
            print("-", warning)

    print("\nModel saved to:", args.output_dir)


if __name__ == "__main__":
    main()
