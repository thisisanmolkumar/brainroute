import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from fastapi.encoders import jsonable_encoder
from starlette.concurrency import run_in_threadpool

from backend.api.dependencies import get_execution_service
from backend.routing.api_schemas import (
    ManualExecuteRequest,
    ManualExecutionResponse,
    RecommendRequest,
)
from backend.routing.runtime import get_routing_service
from backend.routing.schemas import RoutingDecision
from backend.routing.security import (
    require_execution_authorization,
)
from backend.routing.service import RoutingService
from backend.schemas.llm import LLMRequest
from backend.services.execution_service import ExecutionService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/router",
    tags=["BrainRoute Routing"],
)


@router.post(
    "/recommend",
    response_model=RoutingDecision,
)
async def recommend_route(
    payload: RecommendRequest,
    service: Annotated[
        RoutingService,
        Depends(get_routing_service),
    ],
) -> RoutingDecision:

    try:
        return await run_in_threadpool(
            service.route,
            payload.prompt,
            payload.candidates,
            payload.policy,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@router.post(
    "/execute",
    response_model=ManualExecutionResponse,
    dependencies=[Depends(require_execution_authorization)],
)
async def execute_routed_task(
    payload: ManualExecuteRequest,
    routing_service: Annotated[
        RoutingService,
        Depends(get_routing_service),
    ],
    execution_service: Annotated[
        ExecutionService,
        Depends(get_execution_service),
    ],
) -> ManualExecutionResponse:

    # ----------------------------------
    # 1. Recalculate routing decision
    # ----------------------------------

    try:
        decision = await run_in_threadpool(
            routing_service.route,
            payload.prompt,
            payload.candidates,
            payload.policy,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # ----------------------------------
    # 2. Reject infeasible requests
    # ----------------------------------

    selected = decision.recommendation

    if selected is None:
        raise HTTPException(
            status_code=409,
            detail="No eligible routing candidate",
        )

    # ----------------------------------
    # 3. Verify explicit approval
    # ----------------------------------

    if (
        selected.model_id != payload.approved_model_id
        or selected.effort != payload.approved_effort
    ):
        raise HTTPException(
            status_code=409,
            detail=("Approved configuration does not match the current recommendation"),
        )

    if (
        decision.status == "review_required"
        and not payload.acknowledge_uncalibrated_predictions
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Routing evidence is insufficient. "
                "Explicit acknowledgement is required."
            ),
        )

    # ----------------------------------
    # 4. Require a cost estimate
    # ----------------------------------

    if selected.predicted_cost_usd is None:
        raise HTTPException(
            status_code=409,
            detail="Cost prediction unavailable",
        )

    if selected.predicted_cost_usd > payload.policy.max_cost_usd:
        raise HTTPException(
            status_code=409,
            detail="Predicted cost exceeds approved threshold",
        )

    # ----------------------------------
    # 5. Reuse Phase 1 execution engine
    # ----------------------------------

    request = LLMRequest(
        prompt=payload.prompt,
        model_id=selected.model_id,
        effort=selected.effort,
        max_output_tokens=payload.max_output_tokens,
        timeout_seconds=payload.timeout_seconds,
    )

    try:
        execution_result = await execution_service.execute(
            request=request,
        )

    except Exception as exc:
        logger.exception("Approved routed execution failed")

        raise HTTPException(
            status_code=502,
            detail=(
                "LLM execution failed. Check persisted "
                "execution status before retrying."
            ),
        ) from exc

    # ----------------------------------
    # 6. Return decision + execution
    # ----------------------------------

    return ManualExecutionResponse(
        decision=decision,
        execution=jsonable_encoder(execution_result),
        predicted_cost_is_hard_cap=False,
        automatic_retry_enabled=False,
    )
