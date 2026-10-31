from backend.datasets.v2_schema import TaskV2
from backend.datasets.v2_verifiers import VERIFIERS

OUTPUT_INSTRUCTIONS = {
    "gsm8k_final_number": (
        "Return your final numeric answer on the last "
        "line using exactly this format:\n"
        "FINAL_ANSWER: <number>"
    ),
    "bbh_boolean": (
        "Return your final Boolean answer on the last "
        "line using exactly one of these formats:\n"
        "FINAL_ANSWER: True\n"
        "FINAL_ANSWER: False"
    ),
}


def assert_collectible(task: TaskV2) -> None:
    if task.verification_status != "ready":
        raise ValueError(f"{task.task_id}: verification is pending")

    if task.verifier not in VERIFIERS:
        raise ValueError(f"{task.task_id}: verifier not registered")

    if not task.reference_answer:
        raise ValueError(f"{task.task_id}: reference answer missing")

    if task.domain == "coding":
        raise ValueError("Coding evaluation requires an isolated execution environment")


def render_execution_prompt(task: TaskV2) -> str:
    assert_collectible(task)

    instruction = OUTPUT_INSTRUCTIONS[task.verifier]

    return f"{task.raw_prompt}\n\n{instruction}"
