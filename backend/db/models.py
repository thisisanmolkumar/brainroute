from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base

# ------------------------------------
# Tasks
# ------------------------------------


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)

    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    category: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ------------------------------------
# Model Configurations
# ------------------------------------


class ModelConfiguration(Base):
    __tablename__ = "model_configs"

    __table_args__ = (UniqueConstraint("provider", "model_id", "effort"),)

    id: Mapped[int] = mapped_column(primary_key=True)

    provider: Mapped[str] = mapped_column(String(64))

    model_id: Mapped[str] = mapped_column(String(128))

    effort: Mapped[str] = mapped_column(String(32))


# ------------------------------------
# Executions
# ------------------------------------


class Execution(Base):
    __tablename__ = "executions"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Relationships
    task_id: Mapped[int] = mapped_column(
        ForeignKey("tasks.id"),
        nullable=False,
    )

    model_config_id: Mapped[int] = mapped_column(
        ForeignKey("model_configs.id"),
        nullable=False,
    )

    # Execution status
    status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
        nullable=False,
    )

    # Reserved for independent task verification
    success: Mapped[bool | None] = mapped_column(
        nullable=True,
    )

    # Model response
    output_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    provider_response_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    service_tier: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    # Token usage
    input_tokens: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    output_tokens: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    cached_input_tokens: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    cache_write_tokens: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    reasoning_tokens: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    # Cost
    cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 12),
        nullable=True,
    )

    pricing_snapshot: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    metrics_warning: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Latency
    latency_ms: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    end_to_end_latency_ms: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    # Error tracking
    error_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(primary_key=True)

    dataset_path: Mapped[str] = mapped_column(Text)
    dataset_sha256: Mapped[str] = mapped_column(String(64))

    status: Mapped[str] = mapped_column(
        String(32),
        default="running",
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    plan_json: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    max_total_cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 12),
        nullable=True,
    )


class BenchmarkAttempt(Base):
    __tablename__ = "benchmark_attempts"

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "benchmark_task_id",
            "model_id",
            "effort",
            "repeat_index",
            name="uq_benchmark_attempt",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    run_id: Mapped[int] = mapped_column(
        ForeignKey("benchmark_runs.id"),
        nullable=False,
    )

    benchmark_task_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    effort: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    execution_id: Mapped[int | None] = mapped_column(
        ForeignKey("executions.id"),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
        nullable=False,
    )

    verification_status: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    correct: Mapped[bool | None] = mapped_column(
        nullable=True,
    )

    predicted_answer: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    repeat_index: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
