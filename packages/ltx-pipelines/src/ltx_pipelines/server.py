import logging
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger("ltx_server")


class ImageConditioning(BaseModel):
    path: str = Field(..., description="Path to conditioning image on the server filesystem")
    frame_idx: int = Field(..., description="Frame index to apply conditioning at")
    strength: float = Field(..., description="Conditioning strength (0.0-1.0)")
    crf: int = Field(DEFAULT_IMAGE_CRF, description="JPEG CRF for image preprocessing")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the desired video")
    seed: int = Field(10, ge=0, description="Random seed for reproducible generation")
    height: int = Field(1024, ge=64, description="Output height in pixels (divisible by 64)")
    width: int = Field(1536, ge=64, description="Output width in pixels (divisible by 64)")
    num_frames: int = Field(121, ge=1, description="Number of frames (must be k*8+1)")
    frame_rate: float = Field(24.0, gt=0, description="Output frame rate")
    enhance_prompt: bool = Field(False, description="Enable automatic prompt enhancement")
    images: list[ImageConditioning] = Field(default_factory=list, description="Optional image conditioning")

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "GenerateRequest":
        if self.height % 64 != 0:
            raise ValueError(f"height must be divisible by 64, got {self.height}")
        if self.width % 64 != 0:
            raise ValueError(f"width must be divisible by 64, got {self.width}")
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError(f"num_frames must satisfy (frames - 1) % 8 == 0, got {self.num_frames}")
        return self


class ServerConfig(BaseModel):
    distilled_checkpoint_path: str
    gemma_root: str
    spatial_upsampler_path: str
    quantization: str | None = None
    offload_mode: OffloadMode = OffloadMode.NONE
    torch_compile: bool = False
    loras: list[str] = Field(default_factory=list)


def _build_quantization(policy_name: str | None) -> QuantizationPolicy | None:
    if policy_name is None:
        return None
    if policy_name == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    if policy_name == "fp8-scaled-mm":
        return QuantizationPolicy.fp8_scaled_mm()
    raise ValueError(f"Unknown quantization policy: {policy_name}")


def _build_loras(loras: list[str]) -> list[LoraPathStrengthAndSDOps]:
    result = []
    for spec in loras:
        parts = spec.split(maxsplit=1)
        path = str(Path(parts[0]).expanduser().resolve())
        strength = float(parts[1]) if len(parts) > 1 else 1.0
        result.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
    return result


class PipelineManager:
    _instance: "PipelineManager | None" = None

    def __init__(self) -> None:
        self._pipeline: DistilledPipeline | None = None
        self._config: ServerConfig | None = None

    @classmethod
    def get_instance(cls) -> "PipelineManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, config: ServerConfig) -> None:
        if self._pipeline is not None:
            return
        self._config = config
        logger.info("Loading DistilledPipeline...")
        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=config.distilled_checkpoint_path,
            gemma_root=config.gemma_root,
            spatial_upsampler_path=config.spatial_upsampler_path,
            loras=_build_loras(config.loras),
            quantization=_build_quantization(config.quantization),
            torch_compile=config.torch_compile,
            offload_mode=config.offload_mode,
        )
        logger.info("Pipeline loaded successfully.")

    @property
    def pipeline(self) -> DistilledPipeline:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._pipeline


def create_app(config: ServerConfig) -> FastAPI:
    manager = PipelineManager.get_instance()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        manager.initialize(config)
        yield

    app = FastAPI(title="LTX-2 Video Generation API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "gpu_available": torch.cuda.is_available()}

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> FileResponse:
        if manager.pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not ready")

        images = [
            ImageConditioningInput(
                path=img.path,
                frame_idx=img.frame_idx,
                strength=img.strength,
                crf=img.crf,
            )
            for img in req.images
        ]

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(req.num_frames, tiling_config)

        logger.info(
            "Generating: prompt='%s...', seed=%d, %dx%d, frames=%d, fps=%.1f",
            req.prompt[:80],
            req.seed,
            req.width,
            req.height,
            req.num_frames,
            req.frame_rate,
        )

        try:
            with torch.inference_mode():
                video_iter, audio = manager.pipeline(
                    prompt=req.prompt,
                    seed=req.seed,
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    frame_rate=req.frame_rate,
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=req.enhance_prompt,
                )
        except Exception as exc:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            encode_video(
                video=video_iter,
                fps=req.frame_rate,
                audio=audio,
                output_path=tmp_path,
                video_chunks_number=video_chunks_number,
            )
        except Exception as exc:
            Path(tmp_path).unlink(missing_ok=True)
            logger.exception("Video encoding failed")
            raise HTTPException(status_code=500, detail=f"Encoding failed: {exc}") from exc

        logger.info("Generation complete, returning video.")
        return FileResponse(
            tmp_path,
            media_type="video/mp4",
            filename="generated.mp4",
            background=lambda: Path(tmp_path).unlink(missing_ok=True),
        )

    return app


def run_server(config: ServerConfig, host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


def _banner() -> None:
    print(
        "\n"
        "  ╔══════════════════════════════════════════════════════╗\n"
        "  ║              LTX-2 Video Generation API              ║\n"
        "  ╚══════════════════════════════════════════════════════╝\n"
    )


def main() -> None:
    import argparse

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="LTX-2 Video Generation API Server")
    parser.add_argument("--distilled-checkpoint-path", type=str, required=True, help="Path to distilled checkpoint .safetensors")
    parser.add_argument("--gemma-root", type=str, required=True, help="Path to Gemma text encoder directory")
    parser.add_argument("--spatial-upsampler-path", type=str, required=True, help="Path to spatial upsampler .safetensors")
    parser.add_argument("--quantization", type=str, default=None, choices=["fp8-cast", "fp8-scaled-mm"], help="Quantization policy")
    parser.add_argument("--offload", type=str, default="none", choices=["none", "cpu", "disk"], help="Weight offloading strategy")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--lora", type=str, action="append", default=[], help="LoRA adapter: PATH [STRENGTH]. Repeat for multiple.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")

    args = parser.parse_args()

    offload_map = {"none": OffloadMode.NONE, "cpu": OffloadMode.CPU, "disk": OffloadMode.DISK}

    config = ServerConfig(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        gemma_root=args.gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        quantization=args.quantization,
        offload_mode=offload_map[args.offload],
        torch_compile=args.compile,
        loras=args.lora,
    )

    _banner()
    run_server(config, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
