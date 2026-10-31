import argparse
from pathlib import Path

from backend.benchmarks.ml_dataset import (
    aggregate_executions,
    load_collection_records,
    write_jsonl,
)
from backend.ml.data import load_ml_examples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare held-out BrainRoute evaluation data"
    )

    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
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

    args = parser.parse_args()

    records = load_collection_records(args.input)

    if any(record.split != args.split for record in records):
        raise ValueError(
            "Input collection contains records from a different original dataset split"
        )

    examples, statistics = aggregate_executions(
        records,
        include_smoke=False,
    )

    if not examples:
        raise ValueError("No terminal evaluation examples found")

    existing = [
        *load_ml_examples(args.train),
        *load_ml_examples(args.validation),
    ]

    existing_groups = {item.metadata.group_key for item in existing}

    new_groups = {item.metadata.group_key for item in examples}

    overlap = existing_groups & new_groups

    if overlap:
        raise ValueError(
            "Evaluation data overlaps training or "
            f"validation groups: {sorted(overlap)[:5]}"
        )

    write_jsonl(
        args.output,
        examples,
    )

    print("\nEvaluation dataset created")
    print("Output:", args.output)
    print("Raw observations:", statistics["input_records"])
    print("Aggregated examples:", len(examples))
    print("Distinct groups:", len(new_groups))


if __name__ == "__main__":
    main()
