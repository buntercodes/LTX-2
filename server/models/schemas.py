"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ImageInput(BaseModel):
    """Image conditioning input."""
    path: str = Field(..., description="Path to image file or URL")
    frame_idx: int = Field(default=0, ge=0, description="Frame index for conditioning")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Conditioning strength")
    crf: int = Field(default=33, ge=0, le=51, description="CRF value for image encoding")


class GenerateRequest(BaseModel):
    """Video generation request (legacy)."""
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="Text prompt for video generation"
    )
    height: int | None = Field(
        default=None, 
        ge=64, 
        le=1088,
        description="Video height (default: 1024 for two-stage, must be multiple of 64)"
    )
    width: int | None = Field(
        default=None, 
        ge=64, 
        le=1920,
        description="Video width (default: 1536 for two-stage, must be multiple of 64)"
    )
    num_frames: int | None = Field(
        default=None, 
        ge=9, 
        le=321,
        description="Number of frames (must be 8k+1, default: 121)"
    )
    frame_rate: float | None = Field(
        default=None, 
        ge=1.0, 
        le=60.0,
        description="Frames per second (default: 24.0)"
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (default: 10)"
    )
    images: list[ImageInput] = Field(
        default_factory=list,
        description="Input images for conditioning"
    )
    enhance_prompt: bool = Field(
        default=False,
        description="Enable automatic prompt enhancement"
    )
    callback_url: str | None = Field(
        default=None,
        description="URL to receive webhook callback when job completes"
    )

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, v: int | None) -> int | None:
        """Validate frame count follows 8k+1 format."""
        if v is not None and (v - 1) % 8 != 0:
            raise ValueError(
                f"num_frames must be 8k+1 format (e.g., 9, 17, 25, 33, ..., 121). Got: {v}"
            )
        return v

    @field_validator("height", "width")
    @classmethod
    def validate_dimensions(cls, v: int | None) -> int | None:
        """Validate dimensions are divisible by 64."""
        if v is not None and v % 64 != 0:
            raise ValueError(f"Dimension must be divisible by 64. Got: {v}")
        return v


class TextToVideoRequest(BaseModel):
    """Text-to-video generation request."""
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="Text prompt for video generation"
    )
    height: int | None = Field(
        default=None, 
        ge=64, 
        le=1088,
        description="Video height (default: 1024 for two-stage, must be multiple of 64)"
    )
    width: int | None = Field(
        default=None, 
        ge=64, 
        le=1920,
        description="Video width (default: 1536 for two-stage, must be multiple of 64)"
    )
    num_frames: int | None = Field(
        default=None, 
        ge=9, 
        le=321,
        description="Number of frames (must be 8k+1, default: 121)"
    )
    frame_rate: float | None = Field(
        default=None, 
        ge=1.0, 
        le=60.0,
        description="Frames per second (default: 24.0)"
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (default: 10)"
    )
    enhance_prompt: bool = Field(
        default=False,
        description="Enable automatic prompt enhancement"
    )
    callback_url: str | None = Field(
        default=None,
        description="URL to receive webhook callback when job completes"
    )

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, v: int | None) -> int | None:
        """Validate frame count follows 8k+1 format."""
        if v is not None and (v - 1) % 8 != 0:
            raise ValueError(
                f"num_frames must be 8k+1 format (e.g., 9, 17, 25, 33, ..., 121). Got: {v}"
            )
        return v

    @field_validator("height", "width")
    @classmethod
    def validate_dimensions(cls, v: int | None) -> int | None:
        """Validate dimensions are divisible by 64."""
        if v is not None and v % 64 != 0:
            raise ValueError(f"Dimension must be divisible by 64. Got: {v}")
        return v


class ImageToVideoRequest(BaseModel):
    """Image-to-video generation request."""
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="Text prompt for video generation"
    )
    images: list[ImageInput] = Field(
        ..., 
        min_length=1,
        description="Input images for conditioning (at least one required)"
    )
    height: int | None = Field(
        default=None, 
        ge=64, 
        le=1088,
        description="Video height (default: 1024 for two-stage, must be multiple of 64)"
    )
    width: int | None = Field(
        default=None, 
        ge=64, 
        le=1920,
        description="Video width (default: 1536 for two-stage, must be multiple of 64)"
    )
    num_frames: int | None = Field(
        default=None, 
        ge=9, 
        le=321,
        description="Number of frames (must be 8k+1, default: 121)"
    )
    frame_rate: float | None = Field(
        default=None, 
        ge=1.0, 
        le=60.0,
        description="Frames per second (default: 24.0)"
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (default: 10)"
    )
    enhance_prompt: bool = Field(
        default=False,
        description="Enable automatic prompt enhancement"
    )
    callback_url: str | None = Field(
        default=None,
        description="URL to receive webhook callback when job completes"
    )

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, v: int | None) -> int | None:
        """Validate frame count follows 8k+1 format."""
        if v is not None and (v - 1) % 8 != 0:
            raise ValueError(
                f"num_frames must be 8k+1 format (e.g., 9, 17, 25, 33, ..., 121). Got: {v}"
            )
        return v

    @field_validator("height", "width")
    @classmethod
    def validate_dimensions(cls, v: int | None) -> int | None:
        """Validate dimensions are divisible by 64."""
        if v is not None and v % 64 != 0:
            raise ValueError(f"Dimension must be divisible by 64. Got: {v}")
        return v


class GenerateResponse(BaseModel):
    """Video generation response."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(default="Job queued successfully", description="Status message")
    created_at: datetime = Field(default_factory=datetime.now, description="Job creation time")
    estimated_seconds: float | None = Field(
        default=None,
        description="Estimated processing time in seconds"
    )


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    created_at: datetime = Field(..., description="Job creation time")
    started_at: datetime | None = Field(default=None, description="Processing start time")
    completed_at: datetime | None = Field(default=None, description="Completion time")
    error: str | None = Field(default=None, description="Error message if failed")
    result_url: str | None = Field(default=None, description="Download URL when completed")
    metadata: dict | None = Field(default=None, description="Additional job metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Server status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    active_jobs: int = Field(..., description="Number of active jobs")
    queue_size: int = Field(..., description="Number of pending jobs")
    gpu_memory_used: float | None = Field(default=None, description="GPU memory used (GB)")
    gpu_memory_total: float | None = Field(default=None, description="Total GPU memory (GB)")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: dict | None = Field(default=None, description="Additional error details")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    checkpoint_path: str = Field(..., description="Model checkpoint path")
    spatial_upsampler_path: str = Field(..., description="Upsampler path")
    lora_count: int = Field(..., description="Number of loaded LoRAs")
    quantization: str | None = Field(default=None, description="Quantization policy")
    compile_enabled: bool = Field(..., description="torch.compile enabled")
    offload_mode: str = Field(..., description="Weight offloading mode")
    default_resolution: str = Field(..., description="Default resolution (WxH)")
    max_resolution: str = Field(..., description="Maximum resolution (WxH)")
    supported_frame_counts: list[int] = Field(
        ..., 
        description="Example supported frame counts"
    )
