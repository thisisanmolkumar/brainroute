from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.api.dependencies import get_registry
from backend.core.model_registry import (
    ModelConfig,
    ModelRegistry,
)

router = APIRouter(
    prefix="/models",
    tags=["Models"],
)


@router.get(
    "",
    response_model=list[ModelConfig],
)
def list_models(
    registry: Annotated[
        ModelRegistry,
        Depends(get_registry),
    ],
    enabled_only: bool = True,
):
    return registry.list_models(
        enabled_only=enabled_only,
    )


@router.get(
    "/{model_id}",
    response_model=ModelConfig,
)
def get_model(
    model_id: str,
    registry: Annotated[
        ModelRegistry,
        Depends(get_registry),
    ],
):
    try:
        return registry.get_model(model_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
