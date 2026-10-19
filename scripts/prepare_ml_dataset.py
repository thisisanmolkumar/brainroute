import argparse
from pathlib import Path

from backend.benchmarks.ml_dataset import build_ml_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare BrainRoute ML training data")

    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="One or more collection JSONL exports",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/ml/v1"),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke tasks for pipeline testing only",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    manifest = build_ml_dataset(
        input_paths=args.input,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        include_smoke=args.include_smoke,
    )

    print("\n========== BRAINROUTE ML DATASET ==========")

    statistics = manifest["statistics"]

    print(
        "Input records:",
        statistics["input_records"],
    )

    print(
        "Terminal records used:",
        statistics["terminal_records_used"],
    )

    print(
        "Excluded smoke records:",
        statistics["excluded_smoke"],
    )

    print(
        "Excluded nonterminal records:",
        statistics["excluded_nonterminal"],
    )

    print(
        "Aggregated ML examples:",
        statistics["aggregated_examples"],
    )

    print("\nDATASET SPLITS")

    for split, details in manifest["splits"].items():
        print(
            f"{split:12} "
            f"examples={details['examples']:5} "
            f"tasks={details['tasks']:5} "
            f"groups={details['groups']:5}"
        )

    print("\nTARGET COVERAGE")

    print(
        "Complete cost targets:",
        statistics["complete_cost_targets"],
    )

    print(
        "Complete latency targets:",
        statistics["complete_latency_targets"],
    )

    print("\nDataset saved to:", args.output_dir)


if __name__ == "__main__":
    main()
