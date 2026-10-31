import argparse
import json
from pathlib import Path

from backend.datasets.v2_builder import compile_dataset


def main():
    parser = argparse.ArgumentParser(
        description=("Compile and validate BrainRoute Dataset v2")
    )

    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/v2/compiled"),
    )

    parser.add_argument(
        "--existing-ml",
        nargs="*",
        type=Path,
        default=[],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    manifest = compile_dataset(
        input_paths=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        existing_ml_paths=args.existing_ml,
    )

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
