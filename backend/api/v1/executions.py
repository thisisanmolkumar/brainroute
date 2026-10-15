from decimal import Decimal
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
)
from sqlalchemy.exc import SQLAlchemyError

from backend.api.dependencies import (
    get_execution_service,
    get_registry,
    get_repository,
)
from backend.core.config import get_settings
from backend.core.model_registry import ModelRegistry
from backend.db.repositories import ExecutionRepository
from backend.providers.exceptions import (
    LLMProviderError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from backend.schemas.api import (
    ExecuteRequest,
    ExecutionDetail,
    ExecutionHistoryPage,
)
from backend.schemas.execution import ExecutionResult
from backend.schemas.llm import LLMRequest
from backend.services.execution_service import ExecutionService

router = APIRouter(
    tags=["Executions"],
)


# ----------------------------------------
# Budget Validation
# ----------------------------------------


def validate_budget(
    request: ExecuteRequest,
    registry: ModelRegistry,
) -> None:

    try:
        model = registry.validate_configuration(
            request.model_id,
            request.effort,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if model.provider != "openai":
        raise HTTPException(
            status_code=400,
            detail="Only OpenAI execution is supported currently",
        )

    pricing = model.pricing

    # Conservative approximation of prompt tokens:
    # UTF-8 byte count plus an overhead allowance.
    # This is NOT an exact tokenizer or billing guarantee.

    estimated_input_tokens = len(request.prompt.encode("utf-8")) + 512

    max_short = pricing.short_context_max_input_tokens

    if max_short is not None and estimated_input_tokens > max_short:
        raise HTTPException(
            status_code=400,
            detail="Request exceeds supported pricing range",
        )

    # Use the highest configured input-token rate
    # for this preliminary budget estimate.

    input_rates = [
        pricing.input_per_1m_tokens,
    ]

    if pricing.cached_input_per_1m_tokens is not None:
        input_rates.append(pricing.cached_input_per_1m_tokens)

    if pricing.cache_write_per_1m_tokens is not None:
        input_rates.append(pricing.cache_write_per_1m_tokens)

    max_input_rate = max(input_rates)

    estimated_cost = (
        Decimal(estimated_input_tokens) * max_input_rate
        + Decimal(request.max_output_tokens) * pricing.output_per_1m_tokens
    ) / Decimal(1000000)

    # Application-wide maximum budget
    system_budget = Decimal(str(get_settings().max_execution_cost_usd))

    # A request may impose a stricter budget.
    budget = system_budget

    if request.max_cost_usd is not None:
        budget = min(
            system_budget,
            request.max_cost_usd,
        )

    if estimated_cost > budget:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Estimated execution cost exceeds budget",
                "estimated_cost_usd": str(estimated_cost),
                "budget_usd": str(budget),
            },
        )


# ----------------------------------------
# Execute Model
# ----------------------------------------


@router.post(
    "/execute",
    response_model=ExecutionResult,
)
async def execute_model(
    request: ExecuteRequest,
    service: Annotated[
        ExecutionService,
        Depends(get_execution_service),
    ],
    registry: Annotated[
        ModelRegistry,
        Depends(get_registry),
    ],
):

    validate_budget(request, registry)

    # Remove API-only fields before passing the
    # request to our provider-agnostic service.

    llm_request = LLMRequest(
        **request.model_dump(
            exclude={
                "category",
                "max_cost_usd",
            }
        )
    )

    try:
        result = await service.execute(
            request=llm_request,
            category=request.category,
        )

        return result

    except ProviderConfigurationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ProviderAuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ProviderRateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
        ) from exc

    except ProviderTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=str(exc),
        ) from exc

    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from exc


# ----------------------------------------
# List Execution History
# ----------------------------------------


@router.get(
    "/executions",
    response_model=ExecutionHistoryPage,
)
def list_executions(
    repository: Annotated[
        ExecutionRepository,
        Depends(get_repository),
    ],
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 20,
    offset: Annotated[
        int,
        Query(ge=0),
    ] = 0,
):

    try:
        return repository.list_executions(
            limit=limit,
            offset=offset,
        )

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from exc


# ----------------------------------------
# Get Execution Details
# ----------------------------------------


@router.get(
    "/executions/{execution_id}",
    response_model=ExecutionDetail,
)
def get_execution(
    execution_id: Annotated[
        int,
        Path(ge=1),
    ],
    repository: Annotated[
        ExecutionRepository,
        Depends(get_repository),
    ],
):

    try:
        execution = repository.get_execution(
            execution_id,
        )

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from exc

    if execution is None:
        raise HTTPException(
            status_code=404,
            detail=f"Execution {execution_id} not found",
        )

    return execution
