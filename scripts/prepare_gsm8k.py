import argparse
import random
from pathlib import Path

from backend.benchmarks.schema import BenchmarkTask
from datasets import load_dataset


def extract_answer(answer: str) -> str:
    """
    Extract the numeric final answer from GSM8K.
    """

    if "####" not in answer:
        raise ValueError("GSM8K answer does not contain final answer marker")

    final_answer = answer.rsplit("####", 1)[-1].strip()

    # Normalize thousands separators
    final_answer = final_answer.replace(",", "")

    if not final_answer:
        raise ValueError("Empty GSM8K reference answer")

    return final_answer


def prepare_gsm8k(
    split: str,
    limit: int,
    seed: int,
    output: Path,
) -> None:

    print("Loading GSM8K...")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=split,
    )

    if limit < 1 or limit > len(dataset):
        raise ValueError(f"limit must be between 1 and {len(dataset)}")

    # Deterministic sampling
    rng = random.Random(seed)

    indices = sorted(
        rng.sample(
            range(len(dataset)),
            limit,
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", encoding="utf-8") as file:
        for index in indices:
            example = dataset[index]

            task = BenchmarkTask(
                task_id=f"gsm8k_main_{split}_{index:05d}",
                domain="math",
                source="gsm8k",
                split=split,
                prompt=(
                    "Solve the following mathematical problem.\n"
                    "On the final line, write exactly:\n"
                    "FINAL_ANSWER: <number>\n"
                    "Do not include units in the final answer.\n\n"
                    f"{example['question']}"
                ),
                reference_answer=extract_answer(example["answer"]),
                verifier="numeric_final",
                metadata={
                    "dataset": "openai/gsm8k",
                    "original_index": index,
                    "seed": seed,
                },
            )

            file.write(task.model_dump_json() + "\n")

    print(f"Saved {limit} tasks to {output}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/processed/gsm8k_train_30.jsonl"),
    )

    args = parser.parse_args()

    prepare_gsm8k(
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
