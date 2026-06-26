"""Health and system info router."""

import time
from fastapi import APIRouter, Depends, Request

from config import Settings, get_settings
from models.schemas import HealthResponse, ModelInfoResponse
from services.job_queue import JobQueue
from services.pipeline_service import PipelineService

router = APIRouter(prefix="/api/v1", tags=["health"])

# Track server start time
_start_time = time.time()


def get_pipeline(request: Request) -> PipelineService:
    """Get pipeline service from app state."""
    return request.app.state.pipeline


def get_job_queue(request: Request) -> JobQueue:
    """Get job queue from app state."""
    return request.app.state.job_queue


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check server health and resource usage.",
)
async def health_check(
    pipeline: PipelineService = Depends(get_pipeline),
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Check server health."""
    gpu_info = pipeline.get_gpu_info()

    return HealthResponse(
        status="healthy",
        model_loaded=pipeline.is_loaded,
        active_jobs=job_queue.active_count,
        queue_size=job_queue.pending_count,
        gpu_memory_used=gpu_info.get("memory_used_gb"),
        gpu_memory_total=gpu_info.get("memory_total_gb"),
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model information",
    description="Get information about the loaded model.",
)
async def model_info(
    settings: Settings = Depends(get_settings),
) -> ModelInfoResponse:
    """Get model information."""
    # Generate example frame counts
    example_frames = [i * 8 + 1 for i in range(1, 20)]  # 9, 17, 25, ..., 153

    return ModelInfoResponse(
        checkpoint_path=settings.checkpoint_path,
        spatial_upsampler_path=settings.spatial_upsampler_path,
        lora_count=len(settings.lora_paths),
        quantization=settings.quantization,
        compile_enabled=settings.enable_compile,
        offload_mode=settings.offload_mode,
        default_resolution=f"{settings.default_width}x{settings.default_height}",
        max_resolution=f"{settings.max_width}x{settings.max_height}",
        supported_frame_counts=example_frames,
    )
