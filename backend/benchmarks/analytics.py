import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean, median

from pydantic import BaseModel, ConfigDict, Field, ValidationError

TERMINAL_STATUSES = {"completed", "error"}

EFFORT_PAIRS = [
    ("medium", "high"),
    ("high", "xhigh"),
]


class BenchmarkObservation(BaseModel):
    """
    One exported benchmark attempt.

    Only includes fields required for analytics.
    Other JSONL fields are ignored.
    """

    model_config = ConfigDict(extra="ignore")

    run_id: int
    dataset_sha256: str

    task_id: str
    domain: str

    model_id: str
    effort: str
    repeat_index: int = Field(ge=0)

    attempt_status: str
    verification_status: str | None = None
    strict_success: bool | None = None

    cost_usd: Decimal | None = Field(
        default=None,
        ge=0,
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )


class ConfigurationMetrics(BaseModel):
    model_id: str
    effort: str

    planned: int
    terminal: int
    pending: int
    uncertain: int

    correct: int
    incorrect: int
    unparseable: int
    skipped: int
    errors: int

    strict_success_rate: float | None
    verified_accuracy: float | None

    known_cost_attempts: int
    total_known_cost_usd: Decimal

    mean_known_cost_usd: Decimal | None
    median_known_cost_usd: Decimal | None
    cost_per_success_usd: Decimal | None

    latency_samples: int

    mean_latency_ms: float | None
    median_latency_ms: float | None

    p90_latency_ms: float | None
    p95_latency_ms: float | None


class DomainMetrics(BaseModel):
    domain: str
    model_id: str
    effort: str

    terminal: int
    correct: int
    strict_success_rate: float | None


class EffortComparison(BaseModel):
    model_id: str

    lower_effort: str
    higher_effort: str

    matched_attempts: int

    improvements: int
    regressions: int

    success_rate_delta_pp: float

    paired_cost_attempts: int
    mean_cost_delta_usd: Decimal | None


class OracleMetrics(BaseModel):
    comparable_cases: int
    solved_cases: int

    oracle_success_rate: float

    # Lower bound: excludes unsolved-case expenditure.
    cheapest_successful_cost_usd: Decimal

    avg_cost_per_solved_case_usd: Decimal | None

    selected_counts: dict[str, int]

    best_fixed_configuration: str
    best_fixed_success_rate: float


class AnalyticsReport(BaseModel):
    run_id: int
    dataset_sha256: str

    total_records: int

    configurations: list[ConfigurationMetrics]
    domains: list[DomainMetrics]

    effort_comparisons: list[EffortComparison]

    pareto_frontier: list[str]

    oracle: OracleMetrics | None

    warnings: list[str]


def load_observations(
    path: str | Path,
) -> list[BenchmarkObservation]:

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    observations = []
    seen = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                observation = BenchmarkObservation.model_validate_json(line)

            except ValidationError as exc:
                raise ValueError(
                    f"Invalid analytics record at {path}:{line_number}"
                ) from exc

            key = (
                observation.task_id,
                observation.model_id,
                observation.effort,
                observation.repeat_index,
            )

            if key in seen:
                raise ValueError(f"Duplicate benchmark observation: {key}")

            seen.add(key)

            status = observation.attempt_status

            if status not in {
                "pending",
                "running",
                "completed",
                "error",
                "uncertain",
            }:
                raise ValueError(f"Unsupported attempt status: {status}")

            if status in TERMINAL_STATUSES:
                if observation.strict_success is None:
                    raise ValueError(f"Terminal attempt {key} has no success label")

                if observation.strict_success != (
                    observation.verification_status == "correct"
                ):
                    raise ValueError(f"Inconsistent success label for {key}")

            elif observation.strict_success is not None:
                raise ValueError(f"Nonterminal attempt {key} has a success label")

            observations.append(observation)

    if not observations:
        raise ValueError("Analytics dataset is empty")

    run_ids = {row.run_id for row in observations}

    if len(run_ids) != 1:
        raise ValueError("Analytics input contains multiple benchmark runs")

    fingerprints = {row.dataset_sha256 for row in observations}

    if len(fingerprints) != 1:
        raise ValueError("Analytics input contains different dataset versions")

    return observations


def decimal_median(
    values: list[Decimal],
) -> Decimal | None:

    if not values:
        return None

    ordered = sorted(values)
    middle = len(ordered) // 2

    if len(ordered) % 2:
        return ordered[middle]

    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def percentile(
    values: list[float],
    probability: float,
) -> float | None:
    """
    Nearest-rank percentile.

    Example:
        P95 = sorted_values[ceil(0.95 * n) - 1]
    """

    if not values:
        return None

    if not 0 < probability <= 1:
        raise ValueError("Invalid percentile")

    ordered = sorted(values)

    index = math.ceil(probability * len(ordered)) - 1

    return ordered[index]


def configuration_name(
    model_id: str,
    effort: str,
) -> str:
    return f"{model_id} / {effort}"


def calculate_configuration_metrics(
    rows: list[BenchmarkObservation],
) -> ConfigurationMetrics:

    if not rows:
        raise ValueError("No observations supplied")

    model_id = rows[0].model_id
    effort = rows[0].effort

    terminal = [row for row in rows if row.attempt_status in TERMINAL_STATUSES]

    correct = sum(row.strict_success is True for row in terminal)

    incorrect = sum(row.verification_status == "incorrect" for row in terminal)

    unparseable = sum(row.verification_status == "unparseable" for row in terminal)

    skipped = sum(row.verification_status == "skipped" for row in terminal)

    errors = sum(row.attempt_status == "error" for row in rows)

    verified_total = correct + incorrect

    strict_success_rate = correct / len(terminal) if terminal else None

    verified_accuracy = correct / verified_total if verified_total else None

    # ----------------------------------
    # Cost statistics
    # ----------------------------------

    costs = [row.cost_usd for row in terminal if row.cost_usd is not None]

    total_cost = sum(costs, Decimal(0))

    mean_cost = total_cost / len(costs) if costs else None

    median_cost = decimal_median(costs)

    # Only calculate complete cost per success.
    cost_per_success = None

    if correct > 0 and len(costs) == len(terminal):
        cost_per_success = total_cost / correct

    # ----------------------------------
    # Latency statistics
    # ----------------------------------

    latencies = [row.latency_ms for row in terminal if row.latency_ms is not None]

    return ConfigurationMetrics(
        model_id=model_id,
        effort=effort,
        planned=len(rows),
        terminal=len(terminal),
        pending=sum(row.attempt_status == "pending" for row in rows),
        uncertain=sum(row.attempt_status in {"uncertain", "running"} for row in rows),
        correct=correct,
        incorrect=incorrect,
        unparseable=unparseable,
        skipped=skipped,
        errors=errors,
        strict_success_rate=strict_success_rate,
        verified_accuracy=verified_accuracy,
        known_cost_attempts=len(costs),
        total_known_cost_usd=total_cost,
        mean_known_cost_usd=mean_cost,
        median_known_cost_usd=median_cost,
        cost_per_success_usd=cost_per_success,
        latency_samples=len(latencies),
        mean_latency_ms=(mean(latencies) if latencies else None),
        median_latency_ms=(median(latencies) if latencies else None),
        p90_latency_ms=percentile(
            latencies,
            0.90,
        ),
        p95_latency_ms=percentile(
            latencies,
            0.95,
        ),
    )


def calculate_domain_metrics(
    rows: list[BenchmarkObservation],
) -> list[DomainMetrics]:

    groups = defaultdict(list)

    for row in rows:
        key = (
            row.domain,
            row.model_id,
            row.effort,
        )

        groups[key].append(row)

    results = []

    for (domain, model_id, effort), group in sorted(groups.items()):
        terminal = [row for row in group if row.attempt_status in TERMINAL_STATUSES]

        correct = sum(row.strict_success is True for row in terminal)

        results.append(
            DomainMetrics(
                domain=domain,
                model_id=model_id,
                effort=effort,
                terminal=len(terminal),
                correct=correct,
                strict_success_rate=(correct / len(terminal) if terminal else None),
            )
        )

    return results


def calculate_effort_comparisons(
    rows: list[BenchmarkObservation],
) -> list[EffortComparison]:

    grouped = defaultdict(dict)

    for row in rows:
        if row.attempt_status not in TERMINAL_STATUSES:
            continue

        grouped[(row.model_id, row.effort)][(row.task_id, row.repeat_index)] = row

    models = sorted({row.model_id for row in rows})

    comparisons = []

    for model_id in models:
        for lower_effort, higher_effort in EFFORT_PAIRS:
            lower = grouped.get(
                (model_id, lower_effort),
                {},
            )

            higher = grouped.get(
                (model_id, higher_effort),
                {},
            )

            matched = sorted(set(lower) & set(higher))

            if not matched:
                continue

            improvements = 0
            regressions = 0
            success_delta = 0

            cost_deltas = []

            for key in matched:
                low = lower[key]
                high = higher[key]

                low_success = low.strict_success is True
                high_success = high.strict_success is True

                if not low_success and high_success:
                    improvements += 1

                if low_success and not high_success:
                    regressions += 1

                success_delta += int(high_success) - int(low_success)

                if low.cost_usd is not None and high.cost_usd is not None:
                    cost_deltas.append(high.cost_usd - low.cost_usd)

            comparisons.append(
                EffortComparison(
                    model_id=model_id,
                    lower_effort=lower_effort,
                    higher_effort=higher_effort,
                    matched_attempts=len(matched),
                    improvements=improvements,
                    regressions=regressions,
                    success_rate_delta_pp=(100 * success_delta / len(matched)),
                    paired_cost_attempts=len(cost_deltas),
                    mean_cost_delta_usd=(
                        sum(cost_deltas, Decimal(0)) / len(cost_deltas)
                        if cost_deltas
                        else None
                    ),
                )
            )

    return comparisons


def calculate_pareto_frontier(
    configurations: list[ConfigurationMetrics],
) -> list[str]:

    frontier = []

    for candidate in configurations:
        if (
            candidate.strict_success_rate is None
            or candidate.mean_known_cost_usd is None
            or candidate.mean_latency_ms is None
        ):
            continue

        dominated = False

        for other in configurations:
            if other is candidate:
                continue

            if (
                other.strict_success_rate is None
                or other.mean_known_cost_usd is None
                or other.mean_latency_ms is None
            ):
                continue

            no_worse = (
                other.strict_success_rate >= candidate.strict_success_rate
                and other.mean_known_cost_usd <= candidate.mean_known_cost_usd
                and other.mean_latency_ms <= candidate.mean_latency_ms
            )

            strictly_better = (
                other.strict_success_rate > candidate.strict_success_rate
                or other.mean_known_cost_usd < candidate.mean_known_cost_usd
                or other.mean_latency_ms < candidate.mean_latency_ms
            )

            if no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier.append(
                configuration_name(
                    candidate.model_id,
                    candidate.effort,
                )
            )

    return frontier


def calculate_oracle(
    rows: list[BenchmarkObservation],
    configurations: list[ConfigurationMetrics],
) -> OracleMetrics:

    by_case = defaultdict(list)

    for row in rows:
        key = (
            row.task_id,
            row.repeat_index,
        )

        by_case[key].append(row)

    solved_cases = 0
    total_selected_cost = Decimal(0)

    selected_counts = defaultdict(int)

    for case_rows in by_case.values():
        successful = [row for row in case_rows if row.strict_success is True]

        if not successful:
            continue

        # Complete cost data is guaranteed by
        # the caller before oracle calculation.
        selected = min(
            successful,
            key=lambda row: (
                row.cost_usd,
                row.model_id,
                row.effort,
            ),
        )

        solved_cases += 1
        total_selected_cost += selected.cost_usd

        name = configuration_name(
            selected.model_id,
            selected.effort,
        )

        selected_counts[name] += 1

    best_fixed = min(
        configurations,
        key=lambda config: (
            -config.strict_success_rate,
            config.mean_known_cost_usd,
            config.mean_latency_ms,
            config.model_id,
            config.effort,
        ),
    )

    total_cases = len(by_case)

    return OracleMetrics(
        comparable_cases=total_cases,
        solved_cases=solved_cases,
        oracle_success_rate=(solved_cases / total_cases),
        cheapest_successful_cost_usd=(total_selected_cost),
        avg_cost_per_solved_case_usd=(
            total_selected_cost / solved_cases if solved_cases else None
        ),
        selected_counts=dict(selected_counts),
        best_fixed_configuration=configuration_name(
            best_fixed.model_id,
            best_fixed.effort,
        ),
        best_fixed_success_rate=(best_fixed.strict_success_rate),
    )


def analyze_benchmark(
    rows: list[BenchmarkObservation],
) -> AnalyticsReport:

    if not rows:
        raise ValueError("No benchmark observations supplied")

    if len({row.run_id for row in rows}) != 1:
        raise ValueError("Multiple benchmark runs supplied")

    if len({row.dataset_sha256 for row in rows}) != 1:
        raise ValueError("Different dataset versions supplied")

    # ----------------------------------
    # 1. Group by model and effort
    # ----------------------------------

    groups = defaultdict(list)

    for row in rows:
        groups[(row.model_id, row.effort)].append(row)

    configurations = [
        calculate_configuration_metrics(group) for _, group in sorted(groups.items())
    ]

    # ----------------------------------
    # 2. Domain statistics
    # ----------------------------------

    domains = calculate_domain_metrics(rows)

    # ----------------------------------
    # 3. Effort comparisons
    # ----------------------------------

    effort_comparisons = calculate_effort_comparisons(rows)

    # ----------------------------------
    # 4. Check fair-comparison requirements
    # ----------------------------------

    case_sets = [
        {(row.task_id, row.repeat_index) for row in group} for group in groups.values()
    ]

    same_cases = all(case_set == case_sets[0] for case_set in case_sets)

    all_terminal = all(row.attempt_status in TERMINAL_STATUSES for row in rows)

    all_costs_known = all(row.cost_usd is not None for row in rows)

    all_latencies_known = all(row.latency_ms is not None for row in rows)

    warnings = []

    if not same_cases:
        warnings.append(
            "Configurations were not evaluated on identical task/repetition sets."
        )

    if not all_terminal:
        warnings.append(
            "Some attempts are pending, running, or uncertain. "
            "Aggregate success rates exclude these attempts."
        )

    if not all_costs_known:
        warnings.append(
            "Some execution costs are unknown. Cost statistics may be incomplete."
        )

    if not all_latencies_known:
        warnings.append("Some provider latency measurements are missing.")

    total_cases = len({(row.task_id, row.repeat_index) for row in rows})

    if total_cases < 30:
        warnings.append(
            "Fewer than 30 task/repetition cases were observed. "
            "Treat comparisons as exploratory."
        )

    # ----------------------------------
    # 5. Pareto frontier
    # ----------------------------------

    pareto_frontier = []

    if same_cases and all_terminal and all_costs_known and all_latencies_known:
        pareto_frontier = calculate_pareto_frontier(configurations)

    else:
        warnings.append(
            "Pareto analysis omitted because complete, "
            "comparable accuracy/cost/latency data "
            "is unavailable."
        )

    # ----------------------------------
    # 6. Retrospective oracle
    # ----------------------------------

    oracle = None

    if same_cases and all_terminal and all_costs_known:
        oracle = calculate_oracle(
            rows,
            configurations,
        )

    else:
        warnings.append(
            "Oracle analysis omitted because complete, "
            "comparable execution and cost data "
            "is unavailable."
        )

    # ----------------------------------
    # 7. Build report
    # ----------------------------------

    return AnalyticsReport(
        run_id=rows[0].run_id,
        dataset_sha256=rows[0].dataset_sha256,
        total_records=len(rows),
        configurations=configurations,
        domains=domains,
        effort_comparisons=effort_comparisons,
        pareto_frontier=pareto_frontier,
        oracle=oracle,
        warnings=warnings,
    )
