import argparse
from pathlib import Path

from backend.benchmarks.analytics import (
    analyze_benchmark,
    load_observations,
)


def parse_args():
    parser = argparse.ArgumentParser(description="BrainRoute Benchmark Analytics")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL export from Phase 2.4",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path",
    )

    return parser.parse_args()


def format_percentage(value):
    if value is None:
        return "N/A"

    return f"{value:.2%}"


def format_cost(value):
    if value is None:
        return "N/A"

    return f"${value:.8f}"


def format_latency(value):
    if value is None:
        return "N/A"

    return f"{value:.2f} ms"


def main():
    args = parse_args()

    rows = load_observations(args.input)

    report = analyze_benchmark(rows)

    print("\n========== BRAINROUTE ANALYTICS ==========")

    print("Run ID:", report.run_id)
    print("Records:", report.total_records)
    print("Configurations:", len(report.configurations))

    # ----------------------------------
    # Configuration metrics
    # ----------------------------------

    print("\n========== CONFIGURATION METRICS ==========")

    for config in report.configurations:
        print(
            "\n",
            config.model_id,
            "/",
            config.effort,
        )

        print(
            "Terminal attempts:",
            f"{config.terminal}/{config.planned}",
        )

        print(
            "Strict success rate:",
            format_percentage(config.strict_success_rate),
        )

        print(
            "Verified accuracy:",
            format_percentage(config.verified_accuracy),
        )

        print(
            "Known-cost coverage:",
            f"{config.known_cost_attempts}/{config.terminal}",
        )

        print(
            "Mean known cost:",
            format_cost(config.mean_known_cost_usd),
        )

        print(
            "Cost per success:",
            format_cost(config.cost_per_success_usd),
        )

        print(
            "Median latency:",
            format_latency(config.median_latency_ms),
        )

        print(
            "P95 latency:",
            format_latency(config.p95_latency_ms),
        )

    # ----------------------------------
    # Reasoning effort comparisons
    # ----------------------------------

    print("\n========== REASONING EFFORT ==========")

    for comparison in report.effort_comparisons:
        print(
            f"\n{comparison.model_id}: "
            f"{comparison.lower_effort} -> "
            f"{comparison.higher_effort}"
        )

        print(
            "Matched attempts:",
            comparison.matched_attempts,
        )

        print(
            "Improvements:",
            comparison.improvements,
        )

        print(
            "Regressions:",
            comparison.regressions,
        )

        print(
            "Success delta:",
            f"{comparison.success_rate_delta_pp:+.2f} pp",
        )

        print(
            "Mean additional cost:",
            format_cost(comparison.mean_cost_delta_usd),
        )

    # ----------------------------------
    # Pareto frontier
    # ----------------------------------

    print("\n========== PARETO FRONTIER ==========")

    if report.pareto_frontier:
        for configuration in report.pareto_frontier:
            print("-", configuration)

    else:
        print("Unavailable for this run.")

    # ----------------------------------
    # Oracle baseline
    # ----------------------------------

    print("\n========== ORACLE ROUTING ==========")

    if report.oracle is not None:
        oracle = report.oracle

        print(
            "Oracle success rate:",
            format_percentage(oracle.oracle_success_rate),
        )

        print(
            "Best fixed configuration:",
            oracle.best_fixed_configuration,
        )

        print(
            "Best fixed success rate:",
            format_percentage(oracle.best_fixed_success_rate),
        )

        print(
            "Optimistic solved-case cost:",
            format_cost(oracle.cheapest_successful_cost_usd),
        )

    else:
        print("Unavailable for this run.")

    # ----------------------------------
    # Warnings
    # ----------------------------------

    if report.warnings:
        print("\n========== WARNINGS ==========")

        for warning in report.warnings:
            print("-", warning)

    # ----------------------------------
    # Export JSON
    # ----------------------------------

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        print("\nReport saved:", args.output)


if __name__ == "__main__":
    main()
