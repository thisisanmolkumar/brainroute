import argparse
from pathlib import Path

from backend.ml.semantic_baseline import (
    train_semantic_baseline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BrainRoute semantic XGBoost baseline"
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
        "--baseline-dir",
        type=Path,
        default=Path("models/brainroute_xgb_v1"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/brainroute_semantic_v1"),
    )

    parser.add_argument(
        "--revision",
        required=True,
        help="Pinned Hugging Face model commit SHA",
    )

    parser.add_argument(
        "--pca-dimensions",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    report = train_semantic_baseline(
        train_path=args.train,
        validation_path=args.validation,
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        revision=args.revision,
        seed=args.seed,
        pca_dimensions=args.pca_dimensions,
    )

    print("\n========== BRAINROUTE SEMANTIC BASELINE ==========")

    print(
        "Train groups:",
        report["dataset"]["train_groups"],
    )

    print(
        "Validation groups:",
        report["dataset"]["validation_groups"],
    )

    print(
        "PCA dimensions:",
        report["embedding"]["pca_dimensions"],
    )

    print(
        "Best iteration:",
        report["training"]["best_iteration"],
    )

    print("\nVALIDATION METRICS")

    for name, metrics in report["metrics"].items():
        print(
            f"{name}: "
            f"log_loss={metrics['log_loss']:.6f}, "
            f"brier={metrics['brier_score']:.6f}"
        )

    for warning in report["warnings"]:
        print("WARNING:", warning)

    print("\nSaved model to:", args.output_dir)


if __name__ == "__main__":
    main()
