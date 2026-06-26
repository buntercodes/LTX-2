"""Pipeline service for LTX-2 Distilled inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import torch

from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

from config import Settings

logger = logging.getLogger(__name__)


class PipelineService:
    """Wrapper around DistilledPipeline for server usage."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: DistilledPipeline | None = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the distilled pipeline model."""
        logger.info("Loading DistilledPipeline model...")
        start_time = time.time()

        # Build LoRA list
        loras: list[LoraPathStrengthAndSDOps] = []
        for lora_path in self.settings.lora_paths:
            loras.append(LoraPathStrengthAndSDOps(path=lora_path, strength=1.0))

        # Build quantization policy
        quantization = None
        if self.settings.quantization:
            from ltx_core.quantization import (
                fp8_cast_fuse_rule,
                QuantizationPolicy,
            )

            if self.settings.quantization == "fp8-cast":
                from ltx_core.quantization.fp8_cast import build_policy
                quantization = build_policy(self.settings.checkpoint_path)
            elif self.settings.quantization == "fp8-scaled-mm":
                from ltx_core.quantization.fp8_scaled_mm import build_policy
                quantization = build_policy(self.settings.checkpoint_path)

        # Build compilation config
        compilation_config = None
        if self.settings.enable_compile:
            from ltx_core.model.transformer.compiling import CompilationConfig
            compilation_config = CompilationConfig(mode=self.settings.compile_mode)

        # Build offload mode
        from ltx_pipelines.utils.types import OffloadMode
        offload_mode = OffloadMode(self.settings.offload_mode)

        # Initialize pipeline
        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=self.settings.checkpoint_path,
            gemma_root=self.settings.gemma_root,
            spatial_upsampler_path=self.settings.spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
        )

        self._loaded = True
        elapsed = time.time() - start_time
        logger.info(f"Model loaded successfully in {elapsed:.2f}s")

    def unload_model(self) -> None:
        """Unload the pipeline to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def generate_video(
        self,
        prompt: str,
        output_path: Path,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        seed: int | None = None,
        images: list[tuple[str, int, float, int]] | None = None,
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
    ) -> dict:
        """
        Generate a video using the distilled pipeline.

        Args:
            prompt: Text prompt for generation
            output_path: Path to save the output video
            height: Video height (uses default if None)
            width: Video width (uses default if None)
            num_frames: Number of frames (uses default if None)
            frame_rate: FPS (uses default if None)
            seed: Random seed (uses default if None)
            images: List of tuples (path, frame_idx, strength, crf)
            enhance_prompt: Enable prompt enhancement
            tiling_config: Tiling configuration for memory efficiency

        Returns:
            Dictionary with generation metadata
        """
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")

        # Apply defaults
        height = height or self.settings.default_height
        width = width or self.settings.default_width
        num_frames = num_frames or self.settings.default_num_frames
        frame_rate = frame_rate or self.settings.default_frame_rate
        seed = seed if seed is not None else self.settings.default_seed

        # Ensure dimensions are valid (must be divisible by 64 for two-stage pipeline)
        height = max(64, (height // 64) * 64)
        width = max(64, (width // 64) * 64)

        # Use default tiling config if not provided
        if tiling_config is None:
            tiling_config = TilingConfig.default()

        logger.info(
            f"Generating video: prompt='{prompt[:50]}...', "
            f"resolution={width}x{height}, frames={num_frames}, "
            f"fps={frame_rate}, seed={seed}"
        )

        start_time = time.time()

        # Convert image tuples to ImageConditioningInput objects
        from ltx_pipelines.utils.args import ImageConditioningInput
        image_inputs = [
            ImageConditioningInput(path=path, frame_idx=frame_idx, strength=strength, crf=crf)
            for path, frame_idx, strength, crf in (images or [])
        ]

        # Run pipeline
        video_iterator, audio = self._pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=image_inputs,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )

        # Encode video to file
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        encode_video(
            video=video_iterator,
            fps=frame_rate,
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
        )

        elapsed = time.time() - start_time
        file_size = output_path.stat().st_size if output_path.exists() else 0

        metadata = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "seed": seed,
            "generation_time_seconds": round(elapsed, 2),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
        }

        logger.info(
            f"Video generated in {elapsed:.2f}s, "
            f"size: {metadata['file_size_mb']}MB"
        )

        return metadata

    def get_gpu_info(self) -> dict:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {"available": False}

        device = torch.cuda.current_device()
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(device),
            "memory_used_gb": round(
                torch.cuda.memory_allocated(device) / (1024**3), 2
            ),
            "memory_reserved_gb": round(
                torch.cuda.memory_reserved(device) / (1024**3), 2
            ),
            "memory_total_gb": round(
                torch.cuda.get_device_properties(device).total_memory / (1024**3), 2
            ),
        }
