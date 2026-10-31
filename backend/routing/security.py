import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

execution_key_header = APIKeyHeader(
    name="X-BrainRoute-Execution-Token",
    auto_error=False,
)


def require_execution_authorization(
    supplied_token: str | None = Security(execution_key_header),
) -> None:

    enabled = (
        os.getenv(
            "BRAINROUTE_ENABLE_LIVE_EXECUTION",
            "false",
        ).lower()
        == "true"
    )

    if not enabled:
        raise HTTPException(
            status_code=403,
            detail="Live routing execution is disabled",
        )

    expected_token = os.getenv(
        "BRAINROUTE_ROUTER_EXECUTION_TOKEN",
        "",
    )

    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Execution authorization is not configured",
        )

    if supplied_token is None or not secrets.compare_digest(
        supplied_token,
        expected_token,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid execution token",
        )
