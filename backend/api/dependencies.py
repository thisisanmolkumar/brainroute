from collections.abc import AsyncGenerator
from functools import lru_cache

from fastapi import HTTPException

from backend.core.model_registry import ModelRegistry
from backend.db.repositories import ExecutionRepository
from backend.providers.exceptions import (
    ProviderAuthenticationError,
)
from backend.providers.openai_provider import OpenAIProvider
from backend.services.execution_service import ExecutionService


@lru_cache
def get_registry() -> ModelRegistry:
    return ModelRegistry()


def get_repository() -> ExecutionRepository:
    return ExecutionRepository()


async def get_execution_service() -> AsyncGenerator[
    ExecutionService,
    None,
]:
    registry = get_registry()

    try:
        provider = OpenAIProvider(
            registry=registry,
        )

    except ProviderAuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    try:
        yield ExecutionService(
            provider=provider,
            registry=registry,
            repository=get_repository(),
        )

    finally:
        await provider.close()
