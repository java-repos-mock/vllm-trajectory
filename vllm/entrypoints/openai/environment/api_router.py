# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API router for environment information endpoints."""

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.environment.protocol import (
    EnvironmentHealthResponse,
    EnvironmentResponse,
)
from vllm.entrypoints.openai.environment.serving import OpenAIServingEnvironment
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def environment(request: Request) -> OpenAIServingEnvironment:
    """Get the environment serving instance from the application state."""
    return request.app.state.openai_serving_environment


@router.get(
    "/v1/environment",
    response_model=EnvironmentResponse,
    tags=["Environment"],
    summary="Get environment information",
    description=(
        "Returns detailed information about the vLLM instance environment, "
        "including system info, CUDA/GPU details, Python environment, "
        "and relevant package versions."
    ),
)
async def get_environment_info(raw_request: Request) -> JSONResponse:
    """Get complete environment information for the vLLM instance.

    This endpoint returns detailed system and environment information
    useful for debugging, monitoring, and performance analysis.

    Returns:
        JSONResponse containing EnvironmentResponse data
    """
    try:
        handler = environment(raw_request)
        response = await handler.get_environment_info()
        return JSONResponse(content=response.model_dump())
    except RuntimeError as e:
        logger.error(f"Failed to get environment info: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting environment info: {e}")
        raise HTTPException(
            status_code=500, detail="Internal error collecting environment information"
        )


@router.get(
    "/v1/environment/health",
    response_model=EnvironmentHealthResponse,
    tags=["Environment"],
    summary="Get environment health summary",
    description=(
        "Returns a simplified health check with basic environment status. "
        "Useful for quick health monitoring without full environment details."
    ),
)
async def get_environment_health(raw_request: Request) -> JSONResponse:
    """Get a simplified health check with environment summary.

    This is a lightweight endpoint for monitoring systems that need
    quick status checks without the full environment payload.

    Returns:
        JSONResponse containing EnvironmentHealthResponse data
    """
    try:
        handler = environment(raw_request)
        response = await handler.get_environment_health()
        return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"Error getting environment health: {e}")
        # Return degraded status on error rather than failing
        return JSONResponse(
            content=EnvironmentHealthResponse(
                status="unhealthy",
                vllm_version=None,
                cuda_available=False,
                gpu_count=0,
            ).model_dump()
        )


@router.post(
    "/v1/environment/refresh",
    response_model=EnvironmentResponse,
    tags=["Environment"],
    summary="Refresh environment information",
    description=(
        "Force refresh of cached environment information. "
        "Use this if system configuration has changed and you need updated data."
    ),
)
async def refresh_environment_info(raw_request: Request) -> JSONResponse:
    """Force refresh of the cached environment information.

    This endpoint triggers a fresh collection of environment data,
    updating the cache with current system state.

    Returns:
        JSONResponse containing updated EnvironmentResponse data
    """
    try:
        handler = environment(raw_request)
        response = await handler.refresh_environment_info()
        return JSONResponse(content=response.model_dump())
    except RuntimeError as e:
        logger.error(f"Failed to refresh environment info: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error refreshing environment info: {e}")
        raise HTTPException(
            status_code=500, detail="Internal error refreshing environment information"
        )


def attach_router(app: FastAPI) -> None:
    """Attach the environment router to the FastAPI application.

    Args:
        app: The FastAPI application instance
    """
    app.include_router(router)
    logger.info("Environment API router attached")
