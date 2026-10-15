from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from backend.db.models import (
    Execution,
    ModelConfiguration,
    Task,
)
from backend.db.session import SessionLocal
from backend.providers.exceptions import LLMProviderError
from backend.schemas.api import (
    ExecutionDetail,
    ExecutionHistoryPage,
    ExecutionSummary,
)
from backend.schemas.llm import LLMRequest, LLMResponse
from backend.services.metrics import ExecutionMetrics


def build_execution_summary(
    execution: Execution,
    task: Task,
    config: ModelConfiguration,
) -> dict:
    return {
        "id": execution.id,
        "task_id": task.id,
        "provider": config.provider,
        "model_id": config.model_id,
        "effort": config.effort,
        "status": execution.status,
        "success": execution.success,
        "prompt_preview": task.prompt[:120],
        "input_tokens": execution.input_tokens,
        "output_tokens": execution.output_tokens,
        "cost_usd": execution.cost_usd,
        "latency_ms": execution.latency_ms,
        "end_to_end_latency_ms": (execution.end_to_end_latency_ms),
        "created_at": execution.created_at,
        "finished_at": execution.finished_at,
    }


class ExecutionRepository:
    def create_pending(
        self,
        request: LLMRequest,
        provider: str,
        pricing_snapshot: dict,
        category: str | None = None,
    ) -> int:

        with SessionLocal.begin() as session:
            # 1. Register configuration if necessary.
            # PostgreSQL's ON CONFLICT avoids duplicates
            # when concurrent executions use the same model.

            statement = (
                insert(ModelConfiguration)
                .values(
                    provider=provider,
                    model_id=request.model_id,
                    effort=request.effort,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        "provider",
                        "model_id",
                        "effort",
                    ]
                )
            )

            session.execute(statement)

            config_id = session.scalar(
                select(ModelConfiguration.id).where(
                    ModelConfiguration.provider == provider,
                    ModelConfiguration.model_id == request.model_id,
                    ModelConfiguration.effort == request.effort,
                )
            )

            if config_id is None:
                raise RuntimeError("Could not resolve model configuration")

            # 2. Save incoming task.

            task = Task(
                prompt=request.prompt,
                category=category,
            )

            session.add(task)
            session.flush()

            # 3. Create pending execution.

            execution = Execution(
                task_id=task.id,
                model_config_id=config_id,
                status="pending",
                pricing_snapshot=pricing_snapshot,
            )

            session.add(execution)
            session.flush()

            execution_id = execution.id

        return execution_id

    def complete(
        self,
        execution_id: int,
        response: LLMResponse,
        metrics: ExecutionMetrics,
        metrics_warning: str | None = None,
    ) -> None:

        with SessionLocal.begin() as session:
            execution = session.get(
                Execution,
                execution_id,
            )

            if execution is None:
                raise ValueError(f"Execution {execution_id} not found")

            # Execution outcome
            execution.status = response.status

            # We don't know task correctness yet.
            execution.success = None

            # Provider response
            execution.output_text = response.output_text
            execution.provider_response_id = response.provider_response_id
            execution.service_tier = response.service_tier

            # Token usage
            if response.usage is not None:
                usage = response.usage

                execution.input_tokens = usage.input_tokens
                execution.output_tokens = usage.output_tokens

                execution.cached_input_tokens = usage.cached_input_tokens

                execution.cache_write_tokens = usage.cache_write_tokens

                execution.reasoning_tokens = usage.reasoning_tokens

            # Monetary cost
            if metrics.cost is not None:
                execution.cost_usd = metrics.cost.total_cost_usd

            execution.metrics_warning = metrics_warning

            # Latency
            execution.latency_ms = round(metrics.provider_latency_ms)

            execution.end_to_end_latency_ms = round(metrics.end_to_end_latency_ms)

            execution.finished_at = datetime.now(UTC)

    def fail(
        self,
        execution_id: int,
        error: Exception,
        elapsed_ms: float,
    ) -> None:

        with SessionLocal.begin() as session:
            execution = session.get(
                Execution,
                execution_id,
            )

            if execution is None:
                raise ValueError(f"Execution {execution_id} not found")

            execution.status = "error"
            execution.success = None

            execution.error_type = type(error).__name__

            # Avoid persisting arbitrary exception text,
            # which might contain sensitive information.
            if isinstance(error, LLMProviderError):
                execution.error_message = str(error)
            else:
                execution.error_message = "Unexpected internal execution error"

            execution.end_to_end_latency_ms = round(elapsed_ms)

            execution.finished_at = datetime.now(UTC)

    def list_executions(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> ExecutionHistoryPage:

        with SessionLocal() as session:
            # Total number of executions
            total = session.scalar(select(func.count(Execution.id))) or 0

            # Retrieve execution records
            statement = (
                select(
                    Execution,
                    Task,
                    ModelConfiguration,
                )
                .join(
                    Task,
                    Execution.task_id == Task.id,
                )
                .join(
                    ModelConfiguration,
                    Execution.model_config_id == ModelConfiguration.id,
                )
                .order_by(Execution.id.desc())
                .limit(limit)
                .offset(offset)
            )

            rows = session.execute(statement).all()

            items = [
                ExecutionSummary(
                    **build_execution_summary(
                        execution,
                        task,
                        config,
                    )
                )
                for execution, task, config in rows
            ]

            return ExecutionHistoryPage(
                total=total,
                limit=limit,
                offset=offset,
                items=items,
            )

    def get_execution(
        self,
        execution_id: int,
    ) -> ExecutionDetail | None:

        with SessionLocal() as session:
            statement = (
                select(
                    Execution,
                    Task,
                    ModelConfiguration,
                )
                .join(
                    Task,
                    Execution.task_id == Task.id,
                )
                .join(
                    ModelConfiguration,
                    Execution.model_config_id == ModelConfiguration.id,
                )
                .where(Execution.id == execution_id)
            )

            row = session.execute(statement).one_or_none()

            if row is None:
                return None

            execution, task, config = row

            return ExecutionDetail(
                **build_execution_summary(
                    execution,
                    task,
                    config,
                ),
                prompt=task.prompt,
                category=task.category,
                output_text=execution.output_text,
                provider_response_id=(execution.provider_response_id),
                service_tier=execution.service_tier,
                cached_input_tokens=(execution.cached_input_tokens),
                cache_write_tokens=(execution.cache_write_tokens),
                reasoning_tokens=(execution.reasoning_tokens),
                pricing_snapshot=(execution.pricing_snapshot),
                metrics_warning=(execution.metrics_warning),
                error_type=execution.error_type,
                error_message=execution.error_message,
            )
