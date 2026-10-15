from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
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

    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"))

    model_config_id: Mapped[int] = mapped_column(ForeignKey("model_configs.id"))

    status: Mapped[str] = mapped_column(String(32), default="pending")

    success: Mapped[bool | None] = mapped_column(nullable=True)

    input_tokens: Mapped[int | None] = mapped_column(nullable=True)

    output_tokens: Mapped[int | None] = mapped_column(nullable=True)

    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 8), nullable=True)

    latency_ms: Mapped[int | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
