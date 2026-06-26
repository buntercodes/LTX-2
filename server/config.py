"""Configuration management for LTX-2 Inference Server."""

from __future__ import annotations

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="LTX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload for development")

    # Model paths (required)
    checkpoint_path: str = Field(
        ..., 
        description="Path to distilled model checkpoint (.safetensors)"
    )
    gemma_root: str = Field(
        ..., 
        description="Path to Gemma text encoder directory"
    )
    spatial_upsampler_path: str = Field(
        ..., 
        description="Path to spatial upsampler model (.safetensors)"
    )

    # Optional model paths
    lora_paths: list[str] = Field(
        default_factory=list,
        description="List of LoRA adapter paths"
    )

    # Inference defaults
    default_height: int = Field(default=512, ge=64, description="Default video height (must be multiple of 64)")
    default_width: int = Field(default=768, ge=64, description="Default video width (must be multiple of 64)")
    default_num_frames: int = Field(default=121, ge=9, description="Default frame count")
    default_frame_rate: float = Field(default=24.0, ge=1.0, le=120.0, description="Default FPS")
    default_seed: int = Field(default=42, description="Default random seed")

    # Resource limits
    max_concurrent_jobs: int = Field(
        default=2, 
        ge=1, 
        description="Maximum concurrent video generation jobs"
    )
    max_height: int = Field(default=1088, ge=64, description="Maximum video height (must be multiple of 64)")
    max_width: int = Field(default=1920, ge=64, description="Maximum video width (must be multiple of 64)")
    max_num_frames: int = Field(default=321, ge=9, description="Maximum frame count")

    # File management
    output_dir: Path = Field(
        default=Path("outputs"), 
        description="Directory for generated videos"
    )
    temp_dir: Path = Field(
        default=Path("temp"),
        description="Directory for temporary files"
    )
    max_output_files: int = Field(
        default=100,
        description="Maximum number of output files to keep"
    )
    cleanup_after_hours: float = Field(
        default=24.0,
        description="Hours after which output files are deleted"
    )

    # Optimization
    quantization: str | None = Field(
        default=None,
        description="Quantization policy: fp8-cast or fp8-scaled-mm"
    )
    enable_compile: bool = Field(
        default=False,
        description="Enable torch.compile for faster inference"
    )
    compile_mode: str = Field(
        default="default",
        description="torch.compile mode: default, reduce-overhead, max-autotune"
    )
    offload_mode: str = Field(
        default="none",
        description="Weight offloading: none, cpu, disk"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    enable_request_logging: bool = Field(
        default=True, 
        description="Enable request/response logging"
    )

    # CORS
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins"
    )

    @field_validator("output_dir", "temp_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str | None) -> str | None:
        """Validate quantization policy."""
        if v is not None and v not in ("fp8-cast", "fp8-scaled-mm"):
            raise ValueError(f"Invalid quantization: {v}. Must be fp8-cast or fp8-scaled-mm")
        return v

    @field_validator("offload_mode")
    @classmethod
    def validate_offload_mode(cls, v: str) -> str:
        """Validate offload mode."""
        if v not in ("none", "cpu", "disk"):
            raise ValueError(f"Invalid offload mode: {v}. Must be none, cpu, or disk")
        return v

    @field_validator("compile_mode")
    @classmethod
    def validate_compile_mode(cls, v: str) -> str:
        """Validate compile mode."""
        if v not in ("default", "reduce-overhead", "max-autotune"):
            raise ValueError(f"Invalid compile mode: {v}")
        return v


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
