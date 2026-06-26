"""Generation router for video generation endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from config import Settings, get_settings
from models.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    ImageInput,
    JobStatus,
    JobStatusResponse,
    TextToVideoRequest,
    ImageToVideoRequest,
)
from services.job_queue import Job, JobQueue
from services.pipeline_service import PipelineService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["generation"])


def get_pipeline(request: Request) -> PipelineService:
    """Get pipeline service from app state."""
    return request.app.state.pipeline


def get_job_queue(request: Request) -> JobQueue:
    """Get job queue from app state."""
    return request.app.state.job_queue


@router.post(
    "/generate/text-to-video",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Generate video from text",
    description="Submit a text-to-video generation job to the queue.",
)
async def text_to_video(
    request: TextToVideoRequest,
    background_tasks: BackgroundTasks,
    pipeline: PipelineService = Depends(get_pipeline),
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    """Submit a text-to-video generation job."""
    return await _submit_generation_job(
        request=request,
        background_tasks=background_tasks,
        pipeline=pipeline,
        job_queue=job_queue,
        settings=settings,
    )


@router.post(
    "/generate/image-to-video",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Generate video from image",
    description="Submit an image-to-video generation job to the queue.",
)
async def image_to_video(
    request: ImageToVideoRequest,
    background_tasks: BackgroundTasks,
    pipeline: PipelineService = Depends(get_pipeline),
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    """Submit an image-to-video generation job."""
    return await _submit_generation_job(
        request=request,
        background_tasks=background_tasks,
        pipeline=pipeline,
        job_queue=job_queue,
        settings=settings,
    )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Generate video (legacy endpoint)",
    description="Submit a video generation job. Use /text-to-video or /image-to-video for new integrations.",
    deprecated=True,
)
async def generate_video_legacy(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    pipeline: PipelineService = Depends(get_pipeline),
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    """Submit a video generation job (legacy endpoint)."""
    return await _submit_generation_job(
        request=request,
        background_tasks=background_tasks,
        pipeline=pipeline,
        job_queue=job_queue,
        settings=settings,
    )


async def _submit_generation_job(
    request: GenerateRequest | TextToVideoRequest | ImageToVideoRequest,
    background_tasks: BackgroundTasks,
    pipeline: PipelineService,
    job_queue: JobQueue,
    settings: Settings,
) -> GenerateResponse:
    """Internal function to submit generation job."""
    # Check if model is loaded
    if not pipeline.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization.",
        )

    # Check queue capacity
    if job_queue.pending_count >= settings.max_concurrent_jobs * 2:
        raise HTTPException(
            status_code=429,
            detail="Job queue is full. Please try again later.",
        )

    # Create job
    job = job_queue.create_job(
        callback_url=getattr(request, 'callback_url', None),
        prompt=request.prompt,
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
        frame_rate=request.frame_rate,
        seed=request.seed,
        enhance_prompt=getattr(request, 'enhance_prompt', False),
        has_images=len(getattr(request, 'images', [])) > 0,
    )

    # Generate output path
    output_filename = f"{job.job_id}.mp4"
    output_path = settings.output_dir / output_filename

    # Estimate processing time (rough estimate: ~0.1s per frame)
    num_frames = request.num_frames or settings.default_num_frames
    estimated_seconds = num_frames * 0.1

    # Submit background task
    background_tasks.add_task(
        _process_generation,
        job=job,
        pipeline=pipeline,
        job_queue=job_queue,
        settings=settings,
        output_path=output_path,
        request=request,
    )

    return GenerateResponse(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        message="Job queued successfully",
        created_at=job.created_at,
        estimated_seconds=estimated_seconds,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get job status",
    description="Get the status and progress of a video generation job.",
)
async def get_job_status(
    job_id: str,
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
) -> JobStatusResponse:
    """Get job status by ID."""
    job = job_queue.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Build result URL
    result_url = None
    if job.status == JobStatus.COMPLETED and job.output_path:
        if job.output_path.exists():
            result_url = f"/api/v1/download/{job.job_id}"

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        result_url=result_url,
        metadata=job.metadata,
    )


@router.get(
    "/jobs",
    response_model=list[JobStatusResponse],
    summary="List jobs",
    description="List all jobs with optional status filtering.",
)
async def list_jobs(
    status: JobStatus | None = None,
    limit: int = 50,
    offset: int = 0,
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
) -> list[JobStatusResponse]:
    """List jobs with filtering."""
    jobs = job_queue.list_jobs(status=status, limit=limit, offset=offset)

    return [
        JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
            result_url=f"/api/v1/download/{job.job_id}" if job.status == JobStatus.COMPLETED else None,
            metadata=job.metadata,
        )
        for job in jobs
    ]


@router.get(
    "/download/{job_id}",
    summary="Download generated video",
    description="Download the generated video file.",
    responses={404: {"model": ErrorResponse}},
)
async def download_video(
    job_id: str,
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Download generated video file."""
    job = job_queue.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status}",
        )

    if job.output_path is None or not job.output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Output file not found",
        )

    return FileResponse(
        path=job.output_path,
        filename=f"generated_{job_id}.mp4",
        media_type="video/mp4",
    )


@router.delete(
    "/jobs/{job_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
    summary="Cancel/delete a job",
    description="Cancel a pending job or delete a completed/failed job.",
)
async def delete_job(
    job_id: str,
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
) -> None:
    """Delete a job."""
    job = job_queue.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Delete output file if exists
    if job.output_path and job.output_path.exists():
        job.output_path.unlink()
        logger.info(f"Deleted output file: {job.output_path}")

    # Remove from queue
    del job_queue._jobs[job_id]
    logger.info(f"Job {job_id} deleted")


def _process_generation(
    job: Job,
    pipeline: PipelineService,
    job_queue: JobQueue,
    settings: Settings,
    output_path: Path,
    request: GenerateRequest | TextToVideoRequest | ImageToVideoRequest,
) -> None:
    """Process video generation in background."""
    try:
        # Update job status
        job_queue.update_job(job.job_id, status="processing")

        # Convert image inputs to tuples for pipeline
        images = []
        for img in getattr(request, 'images', []):
            images.append((img.path, img.frame_idx, img.strength, img.crf))

        # Generate video
        metadata = pipeline.generate_video(
            prompt=request.prompt,
            output_path=output_path,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            frame_rate=request.frame_rate,
            seed=request.seed,
            images=images if images else None,
            enhance_prompt=getattr(request, 'enhance_prompt', False),
        )

        # Update job with results
        job_queue.update_job(
            job.job_id,
            status="completed",
            progress=100.0,
            output_path=output_path,
            **metadata,
        )

        logger.info(f"Job {job.job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
        job_queue.update_job(
            job.job_id,
            status="failed",
            error=str(e),
        )

    finally:
        job_queue.release()
